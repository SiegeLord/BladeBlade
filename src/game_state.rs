use crate::controls;
use crate::error::Result;
use crate::sfx::Sfx;
use crate::sprite::Sprite;
use crate::utils::{self, load_bitmap, Vec2D, DT};
use allegro::*;
use allegro_font::*;
use allegro_image::*;
use allegro_primitives::*;
use allegro_ttf::*;
use serde_derive::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::{fmt, path};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Options
{
	pub fullscreen: bool,
	pub width: i32,
	pub height: i32,
	pub play_music: bool,
	pub sfx_volume: f32,
	pub music_volume: f32,

	pub controls: controls::Controls,
}

impl Default for Options
{
	fn default() -> Self
	{
		Self {
			fullscreen: true,
			width: 1024,
			height: 728,
			play_music: true,
			sfx_volume: 1.,
			music_volume: 1.,
			controls: controls::Controls::new(),
		}
	}
}

pub fn load_options(core: &Core) -> Result<Options>
{
	let mut path_buf = path::PathBuf::new();
	if cfg!(feature = "use_user_settings")
	{
		path_buf.push(
			core.get_standard_path(StandardPath::UserSettings)
				.map_err(|_| "Couldn't get standard path".to_string())?,
		);
	}
	path_buf.push("options.cfg");
	if path_buf.exists()
	{
		utils::load_config(path_buf.to_str().unwrap())
	}
	else
	{
		Ok(Default::default())
	}
}

pub fn save_options(core: &Core, options: &Options) -> Result<()>
{
	let mut path_buf = path::PathBuf::new();
	if cfg!(feature = "use_user_settings")
	{
		path_buf.push(
			core.get_standard_path(StandardPath::UserSettings)
				.map_err(|_| "Couldn't get standard path".to_string())?,
		);
	}
	std::fs::create_dir_all(&path_buf).map_err(|_| "Couldn't create directory".to_string())?;
	path_buf.push("options.cfg");
	utils::save_config(path_buf.to_str().unwrap(), &options)
}

pub enum NextScreen
{
	Game,
	Menu,
	Quit,
}

pub struct GameState
{
	pub core: Core,
	pub prim: PrimitivesAddon,
	pub image: ImageAddon,
	pub font: FontAddon,
	pub ttf: TtfAddon,
	pub tick: i64,
	pub paused: bool,
	pub sfx: Sfx,
	pub ui_font: Font,
	pub display_width: f32,
	pub display_height: f32,

	bitmaps: HashMap<String, Bitmap>,
	pub options: Options,
	pub controls: controls::ControlsHandler,
}

impl GameState
{
	pub fn new() -> Result<GameState>
	{
		let core = Core::init()?;
		core.set_app_name("BladeBlade");
		core.set_org_name("SiegeLord");

		let options = load_options(&core)?;
		let prim = PrimitivesAddon::init(&core)?;
		let image = ImageAddon::init(&core)?;
		let font = FontAddon::init(&core)?;
		let ttf = TtfAddon::init(&font)?;
		core.install_keyboard()
			.map_err(|_| "Couldn't install keyboard".to_string())?;
		core.install_mouse()
			.map_err(|_| "Couldn't install mouse".to_string())?;

		let ui_font = ttf
			.load_ttf_font("data/AvQest.ttf", 24, Flag::zero())
			.map_err(|_| "Couldn't load 'data/AvQest.ttf'".to_string())?;

		let sfx = Sfx::new(options.sfx_volume, options.music_volume, &core)?;

		let controls = controls::ControlsHandler::new(options.controls.clone());
		Ok(GameState {
			core: core,
			prim: prim,
			image: image,
			tick: 0,
			bitmaps: HashMap::new(),
			font: font,
			ui_font: ui_font,
			display_width: 0.,
			display_height: 0.,
			ttf: ttf,
			sfx: sfx,
			paused: false,
			options: options,
			controls: controls,
		})
	}

	pub fn cache_bitmap<'l>(&'l mut self, name: &str) -> Result<&'l Bitmap>
	{
		Ok(match self.bitmaps.entry(name.to_string())
		{
			Entry::Occupied(o) => o.into_mut(),
			Entry::Vacant(v) => v.insert(load_bitmap(&self.core, name)?),
		})
	}

	pub fn get_bitmap<'l>(&'l self, name: &str) -> Option<&'l Bitmap>
	{
		self.bitmaps.get(name)
	}

	pub fn time(&self) -> f64
	{
		self.tick as f64 * DT as f64
	}
}
