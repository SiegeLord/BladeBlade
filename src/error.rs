use std::backtrace::Backtrace;
use std::{error, fmt};

pub type Result<T> = std::result::Result<T, Error>;

pub struct Error
{
	message: String,
	inner: Option<Box<dyn error::Error + 'static>>,
	backtrace: Backtrace,
}

impl Error
{
	pub fn new(message: String, inner: Option<Box<dyn error::Error + 'static>>) -> Self
	{
		Self {
			message: message,
			inner: inner,
			backtrace: Backtrace::capture(),
		}
	}
}

impl From<String> for Error
{
	fn from(error: String) -> Self
	{
		Self {
			message: error,
			inner: None,
			backtrace: Backtrace::capture(),
		}
	}
}

impl From<hecs::NoSuchEntity> for Error
{
	fn from(error: hecs::NoSuchEntity) -> Self
	{
		Self {
			message: format!("{}", error),
			inner: Some(Box::new(error)),
			backtrace: Backtrace::capture(),
		}
	}
}

impl From<hecs::ComponentError> for Error
{
	fn from(error: hecs::ComponentError) -> Self
	{
		Self {
			message: format!("{}", error),
			inner: Some(Box::new(error)),
			backtrace: Backtrace::capture(),
		}
	}
}

impl fmt::Display for Error
{
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
	{
		write!(f, "{}", self.message,)?;
		if let Some(ref inner) = self.inner
		{
			write!(f, "\nCause: {}", inner)?;
		}
		write!(f, "\nBacktrace:\n{}", self.backtrace)?;
		Ok(())
	}
}

impl error::Error for Error
{
	fn source(&self) -> Option<&(dyn error::Error + 'static)>
	{
		self.inner.as_ref().map(|e| &**e)
	}
}

impl fmt::Debug for Error
{
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
	{
		write!(f, "{}", self)
	}
}
