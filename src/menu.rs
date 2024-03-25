use crate::error::Result;
use crate::game_state::{self, GameState, NextScreen};
use crate::utils::{load_obj, PI};
use crate::{controls, ui};
use allegro::*;
use allegro_font::*;
use allegro_primitives::*;
use na::{Point2, Point3, Rotation2};
use nalgebra as na;

pub struct Menu
{
	blade_obj: Vec<[Point3<f32>; 2]>,

	subscreens: Vec<ui::SubScreen>,
}

impl Menu
{
	pub fn new(state: &mut GameState) -> Result<Self>
	{
		state.cache_bitmap("data/logo.png")?;

		state.sfx.cache_sample("data/ui1.ogg")?;
		state.sfx.cache_sample("data/ui2.ogg")?;

		Ok(Self {
			blade_obj: load_obj("data/blade.obj")?,
			subscreens: vec![],
		})
	}

	pub fn resize(&mut self, state: &GameState)
	{
		self.subscreens = self.subscreens.iter().map(|s| s.remake(state)).collect();
	}

	pub fn reset(&mut self)
	{
		self.subscreens.clear();
	}

	pub fn logic(&mut self, _state: &mut GameState) -> Result<()>
	{
		Ok(())
	}

	pub fn input(&mut self, event: &Event, state: &mut GameState) -> Result<Option<NextScreen>>
	{
		if let Event::KeyDown {
			keycode: KeyCode::Escape,
			..
		} = event
		{
			if self.subscreens.len() > 1
			{
				state.sfx.play_sound("data/ui2.ogg").unwrap();
				self.subscreens.pop().unwrap();
				return Ok(None);
			}
		}

		if self.subscreens.is_empty()
		{
			let mut open_menu = false;
			match event
			{
				Event::MouseButtonUp { .. } =>
				{
					state.sfx.play_sound("data/ui1.ogg")?;
					open_menu = true;
				}
				Event::KeyUp { keycode, .. } => match *keycode
				{
					KeyCode::Escape =>
					{
						state.sfx.play_sound("data/ui2.ogg")?;
						return Ok(Some(NextScreen::Quit));
					}
					KeyCode::Space =>
					{
						state.sfx.play_sound("data/ui1.ogg")?;
						open_menu = true;
					}
					_ => (),
				},
				_ => (),
			}
			if open_menu
			{
				self.subscreens
					.push(ui::SubScreen::MainMenu(ui::MainMenu::new(
						state.display_width,
						state.display_height,
					)));
			}
			return Ok(None);
		}
		else
		{
			if let Some(action) = self.subscreens.last_mut().unwrap().input(state, event)
			{
				match action
				{
					ui::Action::Forward(subscreen_fn) =>
					{
						self.subscreens.push(subscreen_fn(
							state,
							state.display_width,
							state.display_height,
						));
					}
					ui::Action::Start => return Ok(Some(NextScreen::Game)),
					ui::Action::Quit => return Ok(Some(NextScreen::Quit)),
					ui::Action::Back =>
					{
						self.subscreens.pop().unwrap();
					}
					_ => (),
				}
			}
			return Ok(None);
		}
	}

	pub fn draw(&self, state: &GameState) -> Result<()>
	{
		let num_blades = 35;
		let cx = state.display_width / 2.;
		let cy = state.display_height / 2.;

		let color = Color::from_rgb_f(0.2, 1., 0.2);

		let mut vertices = vec![];
		for i in 1..num_blades + 1
		{
			let r = 200. + i as f32 * 32.;
			let offset = ((i * 17) % num_blades) as f32;
			let speed = 10. + 0.1 * ((i * 37) % num_blades) as f32;

			let theta = 2. * PI * (state.time() as f32 / speed) + offset;
			let theta = theta.rem_euclid(2. * PI);

			for line in &self.blade_obj
			{
				for vtx in line
				{
					let vtx = vtx * 30.;
					let vtx_pos = Point2::new(vtx.x + r, vtx.z);
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

		if self.subscreens.is_empty()
		{
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
				&state.ui_font,
				Color::from_rgb_f(c, c, c),
				cx,
				cy + (bh / 2.).floor() + 16.,
				FontAlign::Centre,
				"Click To Start",
			);
		}
		else
		{
			self.subscreens.last().unwrap().draw(state);
		}
		state.core.draw_text(
			&state.ui_font,
			color,
			32.,
			state.display_height - 16. - state.ui_font.get_line_height() as f32,
			FontAlign::Left,
			&format!("Version: {}", game_state::VERSION),
		);

		Ok(())
	}
}
