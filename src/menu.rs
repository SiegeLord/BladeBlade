use crate::error::Result;
use crate::game_state::{GameState, NextScreen};
use crate::utils::PI;
use allegro::*;
use allegro_font::*;
use allegro_primitives::*;
use na::{Point2, Rotation2};
use nalgebra as na;

pub struct Menu
{
	display_width: f32,
	display_height: f32,

	ui_font: Font,
}

impl Menu
{
	pub fn new(state: &mut GameState, display_width: f32, display_height: f32) -> Result<Self>
	{
		state.cache_bitmap("data/logo.png")?;

		state.sfx.cache_sample("data/ui.ogg")?;

		Ok(Self {
			display_width: display_width,
			display_height: display_height,
			ui_font: state
				.ttf
				.load_ttf_font("data/Energon.ttf", 16, Flag::zero())
				.map_err(|_| "Couldn't load 'data/Energon.ttf'".to_string())?,
		})
	}

	pub fn logic(&mut self, _state: &mut GameState) -> Result<()>
	{
		Ok(())
	}

	pub fn input(&mut self, event: &Event, state: &mut GameState) -> Result<Option<NextScreen>>
	{
		let mut ret = None;
		match event
		{
			Event::MouseButtonDown { .. } =>
			{
				ret = Some(NextScreen::Game);
				state.sfx.play_sound("data/ui.ogg")?;
			}
			Event::KeyDown { keycode, .. } => match *keycode
			{
				KeyCode::Escape =>
				{
					ret = Some(NextScreen::Quit);
				}
				KeyCode::Space =>
				{
					ret = Some(NextScreen::Game);
					state.sfx.play_sound("data/ui.ogg")?;
				}
				_ => (),
			},
			_ => (),
		}
		Ok(ret)
	}

	pub fn draw(&self, state: &GameState) -> Result<()>
	{
		let num_blades = 35;
		let cx = self.display_width / 2.;
		let cy = self.display_height / 2.;

		let color = Color::from_rgb_f(0.2, 1., 0.2);

		let mut vertices = vec![];
		for i in 1..num_blades + 1
		{
			let r = 200. + i as f32 * 32.;
			let offset = ((i * 17) % num_blades) as f32;
			let speed = 10. + 0.1 * ((i * 37) % num_blades) as f32;

			let theta = 2. * PI * (state.time() as f32 / speed) + offset;
			let theta = theta.rem_euclid(2. * PI);

			let pts = [(-5., -5.), (0., 5.), (5., -5.)];
			for i in 0..(pts.len() - 1)
			{
				for (dx, dy) in &pts[i..i + 2]
				{
					let vtx_pos = Point2::new(dx + r, *dy);
					let rot = Rotation2::new(theta);
					let vtx_pos = rot * vtx_pos;

					vertices.push(Vertex {
						x: cx + vtx_pos.x,
						y: cy + vtx_pos.y,
						z: 0.,
						u: 0.,
						v: 0.,
						color: color,
					})
				}
			}

			for i in 0..10
			{
				for j in 0..2
				{
					let theta2 = -0.25 * PI * (i + j) as f32 / 10.;
					let dx = r * (theta2 + theta).cos();
					let dy = r * (theta2 + theta).sin();

					vertices.push(Vertex {
						x: cx + dx,
						y: cy + dy,
						z: 0.,
						u: 0.,
						v: 0.,
						color: color,
					})
				}
			}
		}

		state.prim.draw_prim(
			&vertices[..],
			Option::<&Bitmap>::None,
			0,
			vertices.len() as u32,
			PrimType::LineList,
		);

		let bitmap = state.get_bitmap("data/logo.png").unwrap();
		let bw = bitmap.get_width() as f32;
		let bh = bitmap.get_height() as f32;

		state.core.draw_tinted_bitmap(
			bitmap,
			color,
			(cx - bw / 2.).floor(),
			(cy - bh / 2.).floor(),
			Flag::zero(),
		);

		let c = 0.6 + 0.4 * 0.5 * ((5. * state.core.get_time()).sin() + 1.) as f32;

		state.core.draw_text(
			&self.ui_font,
			Color::from_rgb_f(c, c, c),
			cx,
			cy + (bh / 2.).floor() + 16.,
			FontAlign::Centre,
			"Click To Start",
		);

		Ok(())
	}
}
