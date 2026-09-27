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
    /// Show top terms recorded for a category
    Analytics {
        category: Option<String>,
        #[arg(short = 'n', long, default_value_t = 20)]
        n: usize,
        /// Report every configured category
        #[arg(long)]
        all: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Init => not_implemented("init"),
        Command::Learn { category, source } => {
            let _ = (category, source);
            not_implemented("learn")
        }
        Command::Infer { mode, file } => {
            let _ = (mode, file);
            not_implemented("infer")
        }
        Command::Analytics { category, n, all } => {
            let _ = (category, n, all);
            not_implemented("analytics")
        }
    }
}

fn not_implemented(cmd: &str) {
    eprintln!("postmaster-k: `{cmd}` is not implemented yet");
    std::process::exit(2);
}
