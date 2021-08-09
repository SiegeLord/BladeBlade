use crate::error::Result;
use crate::sfx::Sfx;
use crate::sprite::Sprite;
use crate::utils::{load_bitmap, Vec2D, DT};
use allegro::*;
use allegro_font::*;
use allegro_image::*;
use allegro_primitives::*;
use allegro_ttf::*;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt;

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

	bitmaps: HashMap<String, Bitmap>,
}

impl GameState
{
	pub fn new() -> Result<GameState>
	{
		let core = Core::init()?;
		let prim = PrimitivesAddon::init(&core)?;
		let image = ImageAddon::init(&core)?;
		let font = FontAddon::init(&core)?;
		let ttf = TtfAddon::init(&font)?;
		core.install_keyboard()
			.map_err(|_| "Couldn't install keyboard".to_string())?;
		core.install_mouse()
			.map_err(|_| "Couldn't install mouse".to_string())?;

		let sfx = Sfx::new(&core)?;

		Ok(GameState {
			core: core,
			prim: prim,
			image: image,
			tick: 0,
			bitmaps: HashMap::new(),
			font: font,
			ttf: ttf,
			sfx: sfx,
			paused: false,
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
