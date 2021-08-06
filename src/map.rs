use crate::game_state::GameState;
use crate::utils::{camera_project, mat4_to_transform, projection_transform, Vec3D};
use allegro::*;
use allegro_primitives::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point3, Quaternion, RealField, Rotation3, Unit, Vector3,
	Vector4,
};
use nalgebra as na;
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
					x += 4. * rng.gen_range(-1.0..1.0);
					y += 4. * rng.gen_range(-1.0..1.0);
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
}

pub struct Map;

fn get_ground_from_screen(
	x: f32, y: f32, project: Perspective3<f32>, camera: Isometry3<f32>,
) -> Point3<f32>
{
	let near_point = Point3::new(x, y, -1.);
	let far_point = Point3::new(x, y, 1.);

	let near_unprojected = project.unproject_point(&near_point);
	let far_unprojected = project.unproject_point(&far_point);

	let camera_inv = camera.inverse();

	let start = camera_inv * near_unprojected;
	let end = camera_inv * far_unprojected;

	let dir = (end - start);
	let f = (-start.y) / dir.y;
	start + f * dir
}

impl Map
{
	pub fn new() -> Self
	{
		Self
	}

	pub fn draw(
		&self, state: &GameState, player_pos: Vec3D, display_width: f32, display_height: f32,
	)
	{
		let mut vertices = vec![];

		let vertex1 = GridVertex::new(0);
		let vertex2 = GridVertex::new(1);

		let project = projection_transform(display_width, display_height);

		state
			.core
			.use_projection_transform(&mat4_to_transform(project.into_inner()));

		let theta = -std::f32::consts::PI / 3.;
		let height = 300.;
		let camera = camera_project(player_pos.x, height, player_pos.z + 150., player_pos.z);

		state
			.core
			.use_transform(&mat4_to_transform(camera.to_homogeneous()));

		let theta = std::f32::consts::PI / 3.;
		let c = theta.cos();
		let s = theta.sin();

		let dx = 40. * c + 40.;
		let dy = 40. * s + 40. * s;
		let dx2 = 40. * s;

		let top_left = get_ground_from_screen(-1., 1., project, camera);
		let bottom_left = get_ground_from_screen(-1., -1., project, camera);
		let top_right = get_ground_from_screen(1., 1., project, camera);

		let start_y = (top_left.z / dy) as i32;
		let end_y = (bottom_left.z / dy) as i32;
		let start_x = (top_left.x / dx) as i32;
		let end_x = (top_right.x / dx) as i32;

		for x in start_x - 1..end_x + 1
		{
			for y in start_y - 1..end_y + 2
			{
				let shift_x = x as f32 * dx;
				let shift_y = (x % 2) as f32 * dx2 + y as f32 * dy;

				let f = 1. + 0.5 * (state.tick as f32 / 10.).sin();
				let f = y as f32 / 10.;

				let vertex = vertex1.interpolate(&vertex2, f);
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
			}
		}

		for i in 0..40
		{
			for j in 0..2
			{
				let theta = 12. * std::f32::consts::PI * (i + j) as f32 / 40.;
				let dx = 20. * theta.cos();
				let dz = 20. * theta.sin();
				let dy = theta;

				vertices.push(Vertex {
					x: 5. * dy + player_pos.x + dx,
					y: 5. * dy,
					z: player_pos.z + dz,
					u: 0.,
					v: 0.,
					color: Color::from_rgb_f(1., 1., i as f32 / 40.),
				})
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
