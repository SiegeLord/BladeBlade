use crate::error::Result;
use crate::game_state::GameState;
use crate::utils::{
	camera_project, get_ground_from_screen, mat4_to_transform, projection_transform, random_color,
	ColorExt, Vec3D, DT,
};
use allegro::*;
use allegro_primitives::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point3, Quaternion, RealField, Rotation3, Unit, Vector3,
	Vector4,
};
use nalgebra as na;
use rand::prelude::*;

#[derive(Clone)]
pub struct Position
{
	pub pos: Point3<f32>,
}

#[derive(Clone)]
pub struct Target
{
	pub pos: Option<Point3<f32>>,
}

#[derive(Clone)]
pub struct Drawable;

#[derive(Clone)]
pub struct Stats
{
	pub speed: f32,
}

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
				vn.0 = v1.0 * (1. - f) + v2.0 * f;
				vn.1 = v1.1 * (1. - f) + v2.1 * f;
			}
		}
		new
	}
}

pub struct Map
{
	world: hecs::World,
	player: hecs::Entity,
	player_pos: Point3<f32>,
	project: Perspective3<f32>,
	display_width: f32,
	display_height: f32,
	mouse_state: bool,
}

impl Map
{
	pub fn new(display_width: f32, display_height: f32) -> Self
	{
		let mut world = hecs::World::default();

		let player_pos = Point3::new(0., 0., 0.);
		let player = world.spawn((
			Position { pos: player_pos },
			Drawable,
			Target {
				pos: Some(Point3::new(200., 0., 0.)),
			},
			Stats { speed: 200. },
		));

		Self {
			world: world,
			player: player,
			player_pos: player_pos,
			project: projection_transform(display_width, display_height),
			display_width: display_width,
			display_height: display_height,
			mouse_state: false,
		}
	}

	pub fn logic(&mut self, _state: &mut GameState) -> Result<()>
	{
		for (_, (position, target, stats)) in self
			.world
			.query::<(&mut Position, &mut Target, &Stats)>()
			.iter()
		{
			let pos = position.pos;
			let mut arrived = false;
			if let Some(target) = target.pos
			{
				let dir = target - pos;
				let disp = DT * stats.speed;
				if dir.norm() < disp
				{
					arrived = true;
					position.pos = target;
				}
				else
				{
					position.pos = pos + dir.normalize() * disp;
				}
			}
			if arrived
			{
				target.pos = None;
			}
		}

		if let Ok(player_pos) = self.world.get::<Position>(self.player)
		{
			self.player_pos = player_pos.pos;
		}

		Ok(())
	}

	pub fn input(&mut self, event: &Event, _state: &mut GameState) -> Result<()>
	{
		let mut move_to = None;
		match event
		{
			Event::MouseButtonDown { x, y, .. } =>
			{
				self.mouse_state = true;
				move_to = Some((x, y));
			}
			Event::MouseButtonUp { .. } =>
			{
				self.mouse_state = false;
			}
			Event::MouseAxes { x, y, .. } =>
			{
				if self.mouse_state
				{
					move_to = Some((x, y));
				}
			}
			_ => (),
		}

		if let Some((x, y)) = move_to
		{
			if let Ok(mut target) = self.world.get_mut::<Target>(self.player)
			{
				let fx = -1. + 2. * *x as f32 / self.display_width;
				let fy = -1. + 2. * *y as f32 / self.display_height;
				let camera = self.make_camera();

				let ground_pos = get_ground_from_screen(fx, -fy, self.project, camera);
				target.pos = Some(ground_pos);
			}
		}

		Ok(())
	}

	fn make_camera(&self) -> Isometry3<f32>
	{
		let height = 300.;
		camera_project(
			self.player_pos.x,
			height,
			self.player_pos.z + height / 2.,
			self.player_pos.z,
		)
	}

	pub fn draw(&self, state: &GameState)
	{
		let mut vertices = vec![];
		state.core.set_depth_test(Some(DepthFunction::Less));

		state
			.core
			.use_projection_transform(&mat4_to_transform(self.project.into_inner()));

		let camera = self.make_camera();

		state
			.core
			.use_transform(&mat4_to_transform(camera.to_homogeneous()));

		let theta = std::f32::consts::PI / 3.;
		let c = theta.cos();
		let s = theta.sin();

		let dx = 40. * c + 40.;
		let dy = 40. * s + 40. * s;
		let dx2 = 40. * s;

		let top_left = get_ground_from_screen(-1., 1., self.project, camera);
		let bottom_left = get_ground_from_screen(-1., -1., self.project, camera);
		let top_right = get_ground_from_screen(1., 1., self.project, camera);

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

				let rate = 5;
				//~ let fx = (x % rate) as f32 / (rate - 1) as f32;
				//~ let fy = (y % rate) as f32 / (rate - 1) as f32;

				let seed1 = (x / rate + 10000 * (y / rate)).abs() as u64;
				//~ let seed2 = (x / rate + 1 + 10000 * (y / rate)).abs() as u64;
				//~ let seed3 = (x / rate + 10000 * (y / rate + 1)).abs() as u64;
				//~ let seed4 = (x / rate + 1 + 10000 * (y / rate + 1)).abs() as u64;

				let vertex1 = GridVertex::new(seed1);
				//~ let vertex2 = GridVertex::new(seed2);
				//~ let vertex3 = GridVertex::new(seed3);
				//~ let vertex4 = GridVertex::new(seed4);

				let c1 = random_color(seed1);
				//~ let c2 = random_color(seed2);
				//~ let c3 = random_color(seed3);
				//~ let c4 = random_color(seed4);

				//~ let c12 = c1.interpolate(c2, fx);
				//~ let c34 = c3.interpolate(c4, fx);
				let c = c1; //c12.interpolate(c34, fy);

				//~ let vertex12 = vertex1.interpolate(&vertex2, fx);
				//~ let vertex34 = vertex3.interpolate(&vertex4, fx);
				let vertex = vertex1; //vertex12.interpolate(&vertex34, fy);

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
								color: c,
							})
						}
					}
				}
			}
		}

		for (_, (position, _)) in self.world.query::<(&Position, &Drawable)>().iter()
		{
			let pos = position.pos;

			for i in 0..40
			{
				for j in 0..2
				{
					let theta = 12. * std::f32::consts::PI * (i + j) as f32 / 40.;
					let dx = 20. * (1. - i as f32 / 40.) * theta.cos();
					let dz = 20. * (1. - i as f32 / 40.) * theta.sin();
					let dy = theta;

					vertices.push(Vertex {
						x: pos.x + dx,
						y: 5. * dy,
						z: pos.z + dz,
						u: 0.,
						v: 0.,
						color: Color::from_rgb_f(1., 1., i as f32 / 40.),
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
	}
}
