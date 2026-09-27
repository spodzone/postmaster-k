use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "postmaster-k",
    version,
    about = "Multi-category learning email classifier for procmail"
)]
struct Cli {
    /// Classifier state root (default: ~/.postmaster-k; env POSTMASTER_K_HOME)
    #[arg(long, global = true, value_name = "DIR")]
    home: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Print exactly one word: the winning category (or `unsure`)
    Word,
    /// Print a `X-Postmaster-K-Category:` header line
    Header,
    /// Insert classification headers and emit the message on stdout
    Full,
}

#[derive(Subcommand)]
enum Command {
    /// Scaffold the state directory and a default config.toml
    Init,
    /// Learn messages into a category
    Learn {
        category: String,
        /// Directory of one-message-per-file, or '-' to read one message from stdin
        source: Option<String>,
    },
    /// Classify one message (the procmail hot path)
    Infer {
        #[arg(long = "mode", value_enum, default_value_t = Mode::Word)]
        mode: Mode,
        /// Message file; stdin when omitted
        file: Option<PathBuf>,
    },
    /// Show top terms recorded for a category (or all categories when omitted)
    Analytics {
        /// Category to report; omitted means all categories
        category: Option<String>,
        #[arg(short = 'n', long, default_value_t = 20)]
        n: usize,
        /// Report every configured category (overrides category if present)
        #[arg(long, conflicts_with = "category")]
        all: bool,
    },
    /// Sample a Maildir and grid-search threshold / threshold_factor
    Autocalibrate {
        /// Maildir root: each subdirectory is a category of message files
        maildir: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    let home = match postmaster_k::config::resolve_home(cli.home.as_deref()) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("postmaster-k: {e}");
            std::process::exit(2);
        }
    };

    match cli.command {
        Command::Init => run(postmaster_k::config::init(&home)),
        Command::Learn { category, source } => run(postmaster_k::learn::run(
            &home,
            &category,
            source.as_deref(),
        )),
        Command::Infer { mode, file } => {
            let mode = match mode {
                Mode::Word => postmaster_k::infer::Mode::Word,
                Mode::Header => postmaster_k::infer::Mode::Header,
                Mode::Full => postmaster_k::infer::Mode::Full,
            };
            std::process::exit(postmaster_k::infer::run(&home, mode, file.as_deref()));
        }
        Command::Analytics { category, n, all } => {
            // If no specific category is given and --all not set, default to all categories.
            let (cat_opt, all_flag) = if category.is_none() && !all {
                (None, true)
            } else {
                (category.as_deref(), all)
            };
            run(postmaster_k::analytics::run(&home, cat_opt, n, all_flag))
        }
        Command::Autocalibrate { maildir } => {
            match postmaster_k::autocalibrate::run(&maildir, &home) {
                Ok((t, f)) => {
                    println!(
                        "postmaster-k: best parameters — threshold = {}, factor = {}",
                        t, f
                    );
                }
                Err(e) => eprintln!("postmaster-k: {e}"),
            }
        }
    }
}

fn run(result: Result<String, String>) {
    match result {
        Ok(msg) => println!("{msg}"),
        Err(e) => {
            eprintln!("postmaster-k: {e}");
            std::process::exit(2);
        }
    }
}
