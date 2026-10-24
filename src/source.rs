//! Open one capture and stream its records, keeping what was readable.
//!
//! `ingest._iter_records` / `_iter_prefix` in the Python reducer: a container
//! whose tail is missing ends the stream with a note instead of an error, an
//! error opening the container (bad magic, an empty file) still fails, and a
//! desynchronised record tag still fails.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use csiq::raw::RawReader;
use csiq::{CsiRecord, CsiqError, Reader, Width};

/// Which envelope a segment is decoded from (`ingest.csiq_form` sees the same).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Csiq,
    Raw,
}

impl Kind {
    pub fn parse(kind: &str) -> Result<Self, String> {
        match kind {
            "csiq" => Ok(Kind::Csiq),
            "raw" => Ok(Kind::Raw),
            other => Err(format!("unknown capture kind {other:?}")),
        }
    }
}

type Stream = BufReader<Box<dyn Read + Send>>;

enum Inner {
    Csiq(Reader<Stream>),
    Raw(RawReader<Stream>),
}

/// A record stream that stops at a truncated tail and says why.
pub struct Records {
    inner: Option<Inner>,
    /// The reader's message when the container stopped part-way.
    pub truncation: Option<String>,
}

fn open_stream(path: &Path) -> io::Result<Stream> {
    let file = File::open(path)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let stream: Box<dyn Read + Send> = if name.ends_with(".zst") {
        Box::new(zstd::stream::read::Decoder::new(file)?)
    } else {
        Box::new(file)
    };
    Ok(BufReader::with_capacity(1 << 20, stream))
}

/// True for the errors a container cut mid-record produces.
fn is_truncation(err: &CsiqError) -> bool {
    match err {
        // A `.zst` cut mid-frame surfaces from the decoder as an "incomplete"
        // stream rather than as an EOF; it is the same fact about the file.
        CsiqError::Io(e) => {
            e.kind() == io::ErrorKind::UnexpectedEof
                || e.to_string().to_lowercase().contains("incomplete")
        }
        CsiqError::Malformed(msg) => msg.contains("truncated"),
        _ => false,
    }
}

impl Records {
    /// Open *path*. Errors opening the container are returned here, before any record.
    pub fn open(path: &Path, kind: Kind) -> Result<Self, String> {
        let stream = open_stream(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let inner = match kind {
            Kind::Csiq => Inner::Csiq(
                Reader::new(stream)
                    .map_err(|e| format!("TruncatedCapture or bad container: {e}"))?,
            ),
            // `read_raw(path)` with its default width: the raw header carries none.
            Kind::Raw => Inner::Raw(RawReader::new(stream, Width::Noht)),
        };
        Ok(Records {
            inner: Some(inner),
            truncation: None,
        })
    }

    /// The next record, `Ok(None)` at the end or at a truncated tail.
    pub fn next_record(&mut self) -> Result<Option<CsiRecord>, String> {
        let got = match self.inner.as_mut() {
            None => return Ok(None),
            Some(Inner::Csiq(r)) => r.next_record(),
            Some(Inner::Raw(r)) => r.next_record(),
        };
        match got {
            Ok(Some(rec)) => Ok(Some(rec)),
            Ok(None) => {
                self.inner = None;
                Ok(None)
            }
            Err(e) if is_truncation(&e) => {
                self.truncation = Some(format!("TruncatedCapture: {e}"));
                self.inner = None;
                Ok(None)
            }
            Err(e) => Err(e.to_string()),
        }
    }
}
