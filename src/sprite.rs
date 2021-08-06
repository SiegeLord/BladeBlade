use crate::error::Result;
use crate::game_state::GameState;
use crate::utils::{load_config, Vec2D};
use allegro::*;
use serde_derive::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
struct SpriteDesc
{
	bitmap: String,
	width: i32,
	height: i32,
}

#[derive(Clone)]
pub struct Sprite
{
	desc: SpriteDesc,
}

impl Sprite
{
	pub fn load(sprite: &str, state: &mut GameState) -> Result<Sprite>
	{
		let desc: SpriteDesc = load_config(sprite)?;
		state.cache_bitmap(&desc.bitmap)?;
		Ok(Sprite { desc: desc })
	}

	pub fn draw(&self, pos: Vec2D, variant: i32, tint: Color, state: &GameState)
	{
		let bitmap = state.get_bitmap(&self.desc.bitmap).unwrap();

		let w = self.desc.width as f32;
		let h = self.desc.height as f32;

		state.core.draw_tinted_bitmap_region(
			bitmap,
			tint,
			0.,
			variant as f32 * h,
			w,
			h,
			pos.x - w / 2.,
			pos.y + w / 3. - h,
			Flag::zero(),
		);
	}

	pub fn draw_beam(&self, pos: Vec2D, variant: i32, len: f32, theta: f32, state: &GameState)
	{
		let bitmap = state.get_bitmap(&self.desc.bitmap).unwrap();

		//~ let w = self.desc.width as f32;
		let h = self.desc.height as f32;

		state.core.draw_tinted_scaled_rotated_bitmap_region(
			bitmap,
			0.,
			variant as f32 * h,
			len,
			h,
			Color::from_rgb_f(1., 1., 1.),
			len / 2.,
			h / 2.,
			pos.x,
			pos.y,
			1.,
			1.,
			theta,
			Flag::zero(),
		);
	}
}
