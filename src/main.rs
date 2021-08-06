#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![feature(backtrace)]

mod error;
mod game_state;
mod sfx;
mod sprite;
mod utils;

use crate::error::Result;
use crate::utils::{load_config, world_to_screen, Vec2D, DT};
use allegro::*;
use allegro_dialog::*;
use allegro_primitives::*;
use rand::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::rc::Rc;

use na::{Isometry3, Matrix4, Point3, Quaternion, RealField, Rotation3, Unit, Vector3, Vector4};
use nalgebra as na;

fn projection_transform(dw: f32, dh: f32) -> Matrix4<f32>
{
	Matrix4::new_perspective(dw / dh, f32::pi() / 2., 0.1, 2000.)
}

fn mat4_to_transform(mat: Matrix4<f32>) -> Transform
{
	let mut trans = Transform::identity();
	for i in 0..4
	{
		for j in 0..4
		{
			trans.get_matrix_mut()[j][i] = mat[(i, j)];
		}
	}
	trans
}

fn camera_project(x: f32, y: f32, z: f32, dir: f32) -> Matrix4<f32>
{
	let trans = Matrix4::new_translation(&Vector3::new(-x, -y, -z));
	let rot = Matrix4::from_axis_angle(&Unit::new_normalize(Vector3::new(1., 0., 0.)), -dir);

	rot * trans
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
struct Options
{
	fullscreen: bool,
	width: i32,
	height: i32,
	play_music: bool,
}

#[derive(Debug, Clone)]
struct GridVertex
{
	branches: [Vec<(f32, f32)>; 3],
}

impl GridVertex
{
	pub fn new() -> Self
	{
		let mut branches = [vec![], vec![], vec![]];

		let mut rng = thread_rng();

		for i in 0..3
		{
			let theta = 2. * std::f32::consts::PI * i as f32 / 3.;
			for j in 0..5
			{
				let mut x = j as f32 * 10. * theta.cos();
				let mut y = j as f32 * 10. * theta.sin();
				if j != 0 && j != 4
				{
					x += 6. * rng.gen_range(-1.0..1.0);
					y += 6. * rng.gen_range(-1.0..1.0);
				}
				branches[i].push((x, y));
			}
		}

		Self { branches: branches }
	}

	pub fn interpolate(&self, other: &Self, f: f32) -> Self
	{
		let mut new = self.clone();
		for (bn, (b1, b2)) in new
			.branches
			.iter_mut()
			.zip(self.branches.iter().zip(other.branches.iter()))
		{
			for (vn, (v1, v2)) in bn.iter_mut().zip(b1.iter().zip(b2.iter()))
			{
				vn.0 = v1.0 * f + v2.0 * (1. - f);
				vn.1 = v1.1 * f + v2.1 * (1. - f);
			}
		}
		new
	}

	pub fn draw(&self, state: &game_state::GameState)
	{
		for branch in &self.branches
		{
			state.prim.draw_polyline(
				&branch[..],
				LineJoinType::Round,
				LineCapType::Round,
				Color::from_rgb_f(1., 1., 1.),
				2.,
				0.5,
			);
		}
	}
}

fn real_main() -> Result<()>
{
	let options: Options = load_config("options.cfg")?;

	let mut state = game_state::GameState::new()?;
	//~ if options.play_music
	//~ {
	//~ state.sfx.play_music()?;
	//~ }

	if options.fullscreen
	{
		state.core.set_new_display_flags(FULLSCREEN_WINDOW);
	}

	let display = Display::new(&state.core, options.width, options.height)
		.map_err(|_| "Couldn't create display".to_string())?;

	let timer =
		Timer::new(&state.core, DT as f64).map_err(|_| "Couldn't create timer".to_string())?;

	let queue =
		EventQueue::new(&state.core).map_err(|_| "Couldn't create event queue".to_string())?;
	queue.register_event_source(display.get_event_source());
	queue.register_event_source(
		state
			.core
			.get_keyboard_event_source()
			.expect("Couldn't get keyboard"),
	);
	queue.register_event_source(
		state
			.core
			.get_mouse_event_source()
			.expect("Couldn't get mouse"),
	);
	queue.register_event_source(timer.get_event_source());

	let mut quit = false;
	let mut draw = true;

	let vertex1 = GridVertex::new();
	let vertex2 = GridVertex::new();

	timer.start();
	while !quit
	{
		if draw && queue.is_empty()
		{
			let start = state.core.get_time();
			state.core.set_target_bitmap(Some(display.get_backbuffer()));
			state.core.clear_to_color(Color::from_rgb_f(0., 0., 0.2));

			let mut vertices = vec![];

			state
				.core
				.use_projection_transform(&mat4_to_transform(projection_transform(
					display.get_width() as f32,
					display.get_height() as f32,
				)));

			let camera = camera_project(0., 400., 1600., -std::f32::consts::PI / 3.);

			state.core.use_transform(&mat4_to_transform(camera));

			for x in -30..30
			{
				for y in 0..30
				{
					let mut transform = Transform::identity();

					let theta = std::f32::consts::PI / 3.;
					let c = theta.cos();
					let s = theta.sin();

					let dx = 40. * c + 40.;
					let dy = 40. * s + 40. * s;
					let dx2 = 40. * s;

					let shift_x = x as f32 * dx;
					let shift_y = (x % 2) as f32 * dx2 + y as f32 * dy;

					transform.translate(shift_x, shift_y);

					//~ state.core.use_transform(&transform);

					let f = 1. + 0.5 * (state.tick as f32 / 10.).sin();
					let f = y as f32 / 10.;

					let vertex = vertex1.interpolate(&vertex2, f);

					//~ vertex.draw(&state);

					for branch in &vertex.branches
					{
						for idx in 0..branch.len() - 1
						{
							for (x, y) in &branch[idx..idx + 2]
							{
								vertices.push(Vertex {
									x: x + shift_x,
									y: 0.,
									z: y + shift_y,
									u: 0.,
									v: 0.,
									color: Color::from_rgb_f(1., f, 1. - f),
								})
							}
						}
					}
					//~ println!("{}", vertices.len());
				}
			}
			state.prim.draw_prim(
				&vertices[..],
				Option::<&Bitmap>::None,
				0,
				vertices.len() as u32,
				PrimType::LineList,
			);

			state.core.flip_display();
			let end = state.core.get_time();
			println!("{}", 1. / (end - start));
		}

		let event = queue.wait_for_event();
		match event
		{
			Event::DisplayClose { .. } => quit = true,
			Event::TimerTick { .. } =>
			{
				//~ let start = state.core.get_time();
				//~ let end = state.core.get_time();
				//~ println!("{}", 1. / (end - start));
				state.tick += 1;
				state.sfx.update_sounds()?;
				draw = true;
			}
			_ => (),
		}
	}

	Ok(())
}

fn main()
{
	use std::panic::catch_unwind;

	match catch_unwind(|| real_main().unwrap())
	{
		Err(e) =>
		{
			let err: String = e
				.downcast_ref::<&'static str>()
				.map(|&e| e.to_owned())
				.or_else(|| e.downcast_ref::<String>().map(|e| e.clone()))
				.unwrap_or("Unknown error!".to_owned());

			//~ error!("{}", err);
			show_native_message_box(
				None,
				"Error!",
				"An error has occurred!",
				&err,
				Some("You make me sad."),
				MESSAGEBOX_ERROR,
			);
		}
		Ok(_) => (),
	}
}
