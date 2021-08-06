use crate::game_state::GameState;
use crate::utils::{camera_project, mat4_to_transform, projection_transform};
use allegro::*;
use allegro_primitives::*;
use rand::prelude::*;

#[derive(Debug, Clone)]
struct GridVertex
{
	branches: [Vec<(f32, f32)>; 3],
}

impl GridVertex
{
	pub fn new(seed: u64) -> Self
	{
		let mut branches = [vec![], vec![], vec![]];

		let mut rng = StdRng::seed_from_u64(seed);

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

	pub fn draw(&self, state: &GameState)
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

pub struct Map;

impl Map
{
	pub fn new() -> Self
	{
		Self
	}

	pub fn draw(&self, state: &GameState, display_width: f32, display_height: f32)
	{
		let mut vertices = vec![];

		let vertex1 = GridVertex::new(0);
		let vertex2 = GridVertex::new(1);

		state
			.core
			.use_projection_transform(&mat4_to_transform(projection_transform(
				display_width,
				display_height,
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
	}
}
