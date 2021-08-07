use crate::error::Result;
use crate::game_state::GameState;
use crate::utils::{
	camera_project, get_ground_from_screen, mat4_to_transform, max, min, projection_transform,
	random_color, ColorExt, Vec3D, DT,
};
use allegro::*;
use allegro_primitives::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point2, Point3, Quaternion, RealField, Rotation2, Rotation3,
	Unit, Vector2, Vector3, Vector4,
};
use nalgebra as na;
use rand::prelude::*;

static CELL_SIZE: i32 = 2048;
static CELL_RADIUS: i32 = 2;
static SUPER_GRID: i32 = 8;
static VERTEX_RADIUS: f32 = 64.;

#[derive(Clone)]
pub struct Cell
{
	center: Point2<i32>,
	vertices: Vec<GridVertex>,
}

impl Cell
{
	fn new(center: Point2<i32>, world: &mut hecs::World) -> Self
	{
		let mut rng = thread_rng();
		let world_center =
			Point2::new((center.x * CELL_SIZE) as f32, (center.y * CELL_SIZE) as f32);
		if center.x != 0 || center.y != 0
		{
			let w = CELL_SIZE as f32 / 2. - 100.;

			for _ in 0..3
			{
				let dx = world_center.x + rng.gen_range(-w..w);
				let dy = world_center.y + rng.gen_range(-w..w);

				for y in -1..=1
				{
					for x in -1..=1
					{
						world.spawn((
							Enemy {
								time_to_deaggro: 0.,
								fire_delay: 0.25,
							},
							Position {
								pos: Point3::new(dx + 50. * x as f32, 15., dy + 50. * y as f32),
								dir: 0.,
							},
							TimeToMove { time_to_move: 0. },
							Velocity {
								vel: Vector3::new(0., 0., 0.),
							},
							Drawable {
								kind: DrawKind::Enemy,
							},
							Target { pos: None },
							Collision {
								kind: CollisionKind::Enemy,
							},
							Stats::enemy_stats(),
							Weapon {
								time_to_fire: 0.,
								range: 320.,
							},
							Health { health: 100. },
						));
					}
				}
			}
		}

		let num_vertices = (CELL_SIZE as f32 / VERTEX_RADIUS) as i32 / SUPER_GRID;
		let mut vertices = vec![];

		for y in -num_vertices..num_vertices + 1
		{
			for x in -num_vertices..num_vertices + 1
			{
				let seed = (x + center.x * num_vertices * 2)
					+ (num_vertices * num_vertices) * (y + center.y * num_vertices * 2);
				vertices.push(GridVertex::new(seed as u64));
			}
		}

		Self {
			center: center,
			vertices: vertices,
		}
	}

	pub fn world_center(&self) -> Point3<f32>
	{
		Point3::new(
			(self.center.x * CELL_SIZE) as f32,
			0.,
			(self.center.y * CELL_SIZE) as f32,
		)
	}

	pub fn world_to_cell(pos: &Point3<f32>) -> Point2<i32>
	{
		let sz = CELL_SIZE as f32;
		let x = pos.x + sz / 2.;
		let y = pos.z + sz / 2.;
		Point2::new((x / sz).floor() as i32, (y / sz).floor() as i32)
	}

	pub fn contains(&self, pos: &Point3<f32>) -> bool
	{
		self.center == Cell::world_to_cell(pos)
	}

	pub fn get_vertex(&self, pos: &Point3<f32>) -> Option<GridVertex>
	{
		let num_vertices = (2. * CELL_SIZE as f32 / VERTEX_RADIUS) as i32 / SUPER_GRID + 1;
		let disp = pos - self.world_center();

		let super_radius = VERTEX_RADIUS * SUPER_GRID as f32;

		let x = ((disp.x + CELL_SIZE as f32 / 2.) / super_radius) as i32;
		let y = ((disp.z + CELL_SIZE as f32 / 2.) / super_radius) as i32;
		if x < 0 || x >= num_vertices - 1 || y < 0 || y >= num_vertices - 1
		{
			None
		}
		else
		{
			let fx = ((disp.x + CELL_SIZE as f32 / 2.) - x as f32 * super_radius) / super_radius;
			let fy = ((disp.z + CELL_SIZE as f32 / 2.) - y as f32 * super_radius) / super_radius;

			//~ dbg!(x, y, super_radius, num_vertices);
			let vertex1 = &self.vertices[(y * num_vertices + x) as usize];
			let vertex2 = &self.vertices[(y * num_vertices + x + 1) as usize];
			let vertex3 = &self.vertices[((y + 1) * num_vertices + x) as usize];
			let vertex4 = &self.vertices[((y + 1) * num_vertices + x + 1) as usize];

			let vertex12 = vertex1.interpolate(&vertex2, fx);
			let vertex34 = vertex3.interpolate(&vertex4, fx);
			let vertex = vertex12.interpolate(&vertex34, fy);

			//~ println!("Good!");
			Some(vertex)
		}
	}
}

#[derive(Clone)]
pub struct Position
{
	pub pos: Point3<f32>,
	pub dir: f32,
}

#[derive(Clone)]
pub struct Velocity
{
	pub vel: Vector3<f32>,
}

#[derive(Clone)]
pub struct Target
{
	pub pos: Option<Point3<f32>>,
}

#[derive(Clone)]
pub enum CollisionKind
{
	Player,
	Enemy,
	Bullet,
}

impl CollisionKind
{
	pub fn collides_with(&self, other: &Self) -> bool
	{
		match self
		{
			CollisionKind::Player => match other
			{
				CollisionKind::Player => true,
				CollisionKind::Enemy => true,
				CollisionKind::Bullet => true,
			},
			CollisionKind::Enemy => match other
			{
				CollisionKind::Player => true,
				CollisionKind::Enemy => true,
				CollisionKind::Bullet => false,
			},
			CollisionKind::Bullet => match other
			{
				CollisionKind::Player => true,
				CollisionKind::Enemy => false,
				CollisionKind::Bullet => false,
			},
		}
	}
}

#[derive(Clone)]
pub struct Collision
{
	kind: CollisionKind,
}

#[derive(Clone)]
pub enum DrawKind
{
	Player,
	Enemy,
	Bullet,
	Hit,
}

#[derive(Clone)]
pub struct Drawable
{
	kind: DrawKind,
}

#[derive(Clone)]
pub struct Stats
{
	pub speed: f32,
	pub aggro_range: f32,
	pub close_enough_range: f32,
	pub size: f32,
	pub cast_delay: f32,
	pub skill_duration: f32,
	pub area_of_effect: f32,
	pub spell_damage: f32,
}

impl Stats
{
	fn player_stats() -> Self
	{
		Self {
			speed: 200.,
			aggro_range: 100.,
			close_enough_range: 0.,
			size: 20.,
			cast_delay: 0.5,
			skill_duration: 1.,
			area_of_effect: 100.,
			spell_damage: 10.,
		}
	}

	fn enemy_stats() -> Self
	{
		Self {
			speed: 100.,
			aggro_range: 400.,
			close_enough_range: 300.,
			size: 20.,
			cast_delay: 0.5,
			skill_duration: 1.,
			area_of_effect: 100.,
			spell_damage: 1.,
		}
	}

	fn bullet_stats() -> Self
	{
		Self {
			speed: 200.,
			aggro_range: 0.,
			close_enough_range: 0.,
			size: 10.,
			cast_delay: 0.,
			skill_duration: 0.,
			area_of_effect: 100.,
			spell_damage: 1.,
		}
	}

	fn hit_stats() -> Self
	{
		Self {
			speed: 0.,
			aggro_range: 0.,
			close_enough_range: 0.,
			size: 40.,
			cast_delay: 0.,
			skill_duration: 0.,
			area_of_effect: 100.,
			spell_damage: 1.,
		}
	}
}

#[derive(Clone)]
pub struct Enemy
{
	pub time_to_deaggro: f64,
	pub fire_delay: f32,
}

#[derive(Clone)]
pub struct Bullet
{
	pub damage: f32,
}

#[derive(Clone)]
pub struct Weapon
{
	time_to_fire: f64,
	range: f32,
}

#[derive(Clone)]
pub struct TimeToDie
{
	time_to_die: f64,
}

#[derive(Clone)]
pub struct TimeToMove
{
	time_to_move: f64,
}

#[derive(Clone)]
pub struct BladeBlade
{
	num_blades: i32,
	time_to_fire: f64,
	time_to_lose_blade: f64,
	time_to_hit: f64,
}

#[derive(Clone)]
pub struct Health
{
	health: f32,
}

#[derive(Debug, Clone)]
pub struct GridVertex
{
	branches: [Vec<(f32, f32, f32)>; 3],
	color: Color,
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
				let mut x = j as f32 * VERTEX_RADIUS * theta.cos() / 4.;
				let mut z = j as f32 * VERTEX_RADIUS * theta.sin() / 4.;
				let mut y = 0.;
				if j != 0 && j != 4
				{
					x += 4. * rng.gen_range(-1.0..1.0);
					y += 4. * rng.gen_range(-1.0..1.0);
					z += 4. * rng.gen_range(-1.0..1.0);
				}
				branches[i].push((x, y, z));
			}
		}

		Self {
			branches: branches,
			color: random_color(rng.gen_range(0..16000), 1., 0.5),
		}
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
				vn.2 = v1.2 * (1. - f) + v2.2 * f;
			}
		}
		new.color = self.color.interpolate(other.color, f);
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
	mouse_state: [bool; 10],
	mouse_pos: (i32, i32),
	space_state: bool,

	cells: Vec<Cell>,
}

impl Map
{
	pub fn new(display_width: f32, display_height: f32) -> Self
	{
		let mut world = hecs::World::default();

		let player_pos = Point3::new(0., 15., 0.);
		let player = world.spawn((
			Position {
				pos: player_pos,
				dir: 0.,
			},
			TimeToMove { time_to_move: 0. },
			Velocity {
				vel: Vector3::new(0., 0., 0.),
			},
			Drawable {
				kind: DrawKind::Player,
			},
			Target { pos: None },
			Collision {
				kind: CollisionKind::Player,
			},
			Stats::player_stats(),
			BladeBlade {
				num_blades: 0,
				time_to_fire: 0.,
				time_to_lose_blade: 0.,
				time_to_hit: 0.,
			},
			Health { health: 100. },
		));

		let mut cells = vec![];
		for y in -CELL_RADIUS..=CELL_RADIUS
		{
			for x in -CELL_RADIUS..=CELL_RADIUS
			{
				cells.push(Cell::new(Point2::new(x, y), &mut world));
			}
		}

		Self {
			world: world,
			player: player,
			player_pos: player_pos,
			project: projection_transform(display_width, display_height),
			display_width: display_width,
			display_height: display_height,
			mouse_state: [false; 10],
			mouse_pos: (0, 0),
			space_state: false,
			cells: cells,
		}
	}

	pub fn logic(&mut self, state: &mut GameState) -> Result<()>
	{
		if self.mouse_state[1]
		{
			let (x, y) = self.mouse_pos;
			if let Ok(mut target) = self.world.get_mut::<Target>(self.player)
			{
				let fx = -1. + 2. * x as f32 / self.display_width;
				let fy = -1. + 2. * y as f32 / self.display_height;
				let camera = self.make_camera();

				let ground_pos = get_ground_from_screen(fx, -fy, self.project, camera);
				target.pos = Some(ground_pos);
			}
		}

		// position -> target
		for (_, (position, velocity, target, stats)) in self
			.world
			.query::<(&Position, &mut Velocity, &mut Target, &Stats)>()
			.iter()
		{
			//~ println!("{:?}", id);
			let pos = position.pos;
			let mut arrived = false;
			if let Some(target) = target.pos
			{
				let dir = target - pos;
				let disp = DT * stats.speed;
				let dist = dir.norm();
				if dist < disp
				{
					//~ println!(" arrived");
					arrived = true;
				}
				else if dist > stats.close_enough_range
				{
					//~ println!(" moving to target");
					velocity.vel = dir.normalize() * stats.speed;
				}
				else
				{
					//~ println!(" close enough");
					velocity.vel = Vector3::new(0., 0., 0.);
				}
			}
			else
			{
				//~ println!(" no target");
				velocity.vel = Vector3::new(0., 0., 0.);
			}

			if arrived
			{
				target.pos = None;
				//~ println!(" arrived again");
				velocity.vel = Vector3::new(0., 0., 0.);
			}
		}

		// collision detection
		for (id, (position, velocity, collision, stats)) in self
			.world
			.query::<(&Position, &mut Velocity, &Collision, &Stats)>()
			.iter()
		{
			for (other_id, (other_position, other_collision, other_stats)) in
				self.world.query::<(&Position, &Collision, &Stats)>().iter()
			{
				if id == other_id
				{
					continue;
				}
				if !collision.kind.collides_with(&other_collision.kind)
				{
					continue;
				}
				let min_dist = other_stats.size + stats.size;
				let disp = position.pos - other_position.pos;
				let dist = disp.norm();
				if dist < min_dist
				{
					velocity.vel +=
						200. * disp.normalize()
							* max(2. * ((min_dist - dist) / min_dist).powf(2.), 1.);
				}
			}
		}

		// velocity application
		for (id, (position, velocity)) in self.world.query::<(&mut Position, &Velocity)>().iter()
		{
			if let Ok(time_to_move) = self.world.get::<TimeToMove>(id)
			{
				if time_to_move.time_to_move > state.time()
				{
					continue;
				}
			}
			position.pos += DT * velocity.vel;
		}

		// direction application
		for (_, (position, target)) in self.world.query::<(&mut Position, &Target)>().iter()
		{
			if let Some(pos) = target.pos
			{
				let disp = pos - position.pos;
				if disp.norm() > 0.
				{
					let new_dir = Vector2::new(disp.x, disp.z).normalize();
					let old_dir = Vector2::new(position.dir.cos(), position.dir.sin());
					let f = 0.9;
					let dir = f * old_dir + (1. - f) * new_dir;
					position.dir = dir.y.atan2(dir.x);
				}
			}
		}

		// Aggro
		if let Ok(player_pos) = self.world.get::<Position>(self.player)
		{
			for (_id, (enemy, position, target, stats)) in self
				.world
				.query::<(&mut Enemy, &mut Position, &mut Target, &Stats)>()
				.iter()
			{
				let pos = position.pos;
				let dist = (player_pos.pos - pos).norm();
				if dist < stats.aggro_range || state.time() < enemy.time_to_deaggro
				{
					//~ println!("Aggro {:?}", id);
					target.pos = Some(player_pos.pos);
					if dist < stats.aggro_range
					{
						enemy.time_to_deaggro = state.time() + 5.;
					}
				}
				else if state.time() > enemy.time_to_deaggro
				{
					//~ println!("Deaggro {:?}", id);
					target.pos = None;
				}
			}
		}

		// Fire weapon
		let mut new_bullets = vec![];
		for (_id, (enemy, position, target, weapon, time_to_move, stats)) in self
			.world
			.query::<(
				&Enemy,
				&Position,
				&Target,
				&mut Weapon,
				&mut TimeToMove,
				&Stats,
			)>()
			.iter()
		{
			if let Some(pos) = target.pos
			{
				//~ println!("Fire {:?}", id);
				if state.time() > weapon.time_to_fire
				{
					weapon.time_to_fire =
						state.time() + enemy.fire_delay as f64 + stats.cast_delay as f64;
					time_to_move.time_to_move = state.time() + stats.cast_delay as f64;

					new_bullets.push((position.pos, pos, stats.spell_damage));
				}
			}
		}

		for (start, dest, spell_damage) in new_bullets
		{
			let dir = dest - start;
			let stats = Stats::bullet_stats();
			self.world.spawn((
				Bullet {
					damage: spell_damage,
				},
				Position {
					pos: start,
					dir: dir.z.atan2(dir.x),
				},
				Velocity {
					vel: dir.normalize() * stats.speed,
				},
				Drawable {
					kind: DrawKind::Bullet,
				},
				stats,
				TimeToDie {
					time_to_die: state.time() + 5.,
				},
			));
			//~ println!("Fired: {:?}", id);
		}

		// Bullet to player collision
		let mut to_die = vec![];
		let mut hits = vec![];
		if let (Ok(player_pos), Ok(mut health), Ok(player_stats)) = (
			self.world.get::<Position>(self.player),
			self.world.get_mut::<Health>(self.player),
			self.world.get::<Stats>(self.player),
		)
		{
			for (id, (bullet, position, stats)) in
				self.world.query::<(&Bullet, &Position, &Stats)>().iter()
			{
				let pos = position.pos;
				let dist = (player_pos.pos - pos).norm();
				if dist < stats.size + player_stats.size
				{
					to_die.push(id);
					health.health -= bullet.damage;
					hits.push(player_pos.pos);
				}
			}
		}

		// Player's blade blade
		if let (
			Ok(mut blade_blade),
			Ok(stats),
			Ok(mut time_to_move),
			Ok(mut target),
			Ok(position),
		) = (
			self.world.get_mut::<BladeBlade>(self.player),
			self.world.get::<Stats>(self.player),
			self.world.get_mut::<TimeToMove>(self.player),
			self.world.get_mut::<Target>(self.player),
			self.world.get_mut::<Position>(self.player),
		)
		{
			if self.space_state && state.time() > blade_blade.time_to_fire
			{
				blade_blade.time_to_fire = state.time() + stats.cast_delay as f64;
				blade_blade.time_to_lose_blade = state.time() + stats.skill_duration as f64;
				blade_blade.num_blades = min(10, blade_blade.num_blades + 1);
				time_to_move.time_to_move = state.time() + stats.cast_delay as f64;
				target.pos = None;
			}
			if state.time() > blade_blade.time_to_lose_blade
			{
				blade_blade.time_to_lose_blade = state.time() + stats.skill_duration as f64;
				blade_blade.num_blades = max(0, blade_blade.num_blades - 1);
			}

			if blade_blade.num_blades > 0 && state.time() > blade_blade.time_to_hit
			{
				blade_blade.time_to_hit =
					state.time() + (0.5 / (blade_blade.num_blades as f32).sqrt()) as f64;

				for (_id, (_, enemy_position, enemy_stats, mut health)) in self
					.world
					.query::<(&mut Enemy, &mut Position, &Stats, &mut Health)>()
					.iter()
				{
					let dist = (position.pos - enemy_position.pos).norm();
					if dist < stats.area_of_effect + enemy_stats.size
					{
						hits.push(enemy_position.pos);
						health.health -= stats.spell_damage;
					}
				}
			}
		}

		// Hits
		for position in hits
		{
			self.world.spawn((
				Position {
					pos: position,
					dir: 0.,
				},
				Drawable {
					kind: DrawKind::Hit,
				},
				Stats::hit_stats(),
				TimeToDie {
					time_to_die: state.time() + 0.05,
				},
			));
		}

		// Health
		for (id, health) in self.world.query::<&Health>().iter()
		{
			if health.health < 0.
			{
				to_die.push(id);
			}
		}

		// Time to die
		for (id, time_to_die) in self.world.query::<&TimeToDie>().iter()
		{
			if state.time() > time_to_die.time_to_die
			{
				to_die.push(id);
			}
		}

		// Cell changes
		let mut new_cell_centers = vec![];
		if let Ok(player_pos) = self.world.get::<Position>(self.player)
		{
			let player_cell = Cell::world_to_cell(&player_pos.pos);

			let mut good_cells = vec![];

			for cell in &self.cells
			{
				let disp = cell.center - player_cell;
				if disp.x.abs() > CELL_RADIUS || disp.y.abs() > CELL_RADIUS
				{
					for (id, position) in self.world.query::<&Position>().iter()
					{
						if cell.contains(&position.pos)
						{
							to_die.push(id);
						}
					}
					println!("Killed {}", cell.center);
				}
				else
				{
					good_cells.push(cell.clone())
				}
			}

			self.cells.clear();

			for dy in -CELL_RADIUS..=CELL_RADIUS
			{
				for dx in -CELL_RADIUS..=CELL_RADIUS
				{
					let cell_center = player_cell + Vector2::new(dx, dy);

					let mut found = false;
					for cell in &good_cells
					{
						if cell.center == cell_center
						{
							self.cells.push(cell.clone());
							found = true;
							break;
						}
					}

					if !found
					{
						new_cell_centers.push(cell_center);
						println!("New cell {}", cell_center);
					}
				}
			}
		}
		for cell_center in new_cell_centers
		{
			self.cells.push(Cell::new(cell_center, &mut self.world));
		}

		// Remove dead entities
		for id in to_die
		{
			self.world.despawn(id)?;
		}

		// Camera pos
		if let Ok(player_pos) = self.world.get::<Position>(self.player)
		{
			self.player_pos = player_pos.pos;
		}

		Ok(())
	}

	pub fn input(&mut self, event: &Event, _state: &mut GameState) -> Result<()>
	{
		match event
		{
			Event::MouseButtonDown { button, x, y, .. } =>
			{
				self.mouse_pos = (*x, *y);
				self.mouse_state[*button as usize] = true;
			}
			Event::MouseButtonUp { button, .. } =>
			{
				self.mouse_state[*button as usize] = false;
			}
			Event::MouseAxes { x, y, .. } =>
			{
				self.mouse_pos = (*x, *y);
			}
			Event::KeyDown { keycode, .. } =>
			{
				if *keycode == KeyCode::Space
				{
					self.space_state = true;
				}
			}
			Event::KeyUp { keycode, .. } =>
			{
				if *keycode == KeyCode::Space
				{
					self.space_state = false;
				}
			}
			_ => (),
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

		let dx = VERTEX_RADIUS * c + VERTEX_RADIUS;
		let dy = VERTEX_RADIUS * s + VERTEX_RADIUS * s;
		let dx2 = VERTEX_RADIUS * s;

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

				//~ let rate = 8;
				//~ let fx = (x % rate) as f32 / rate as f32;
				//~ let fy = (y % rate) as f32 / rate as f32;

				//~ let seed1 = (x / rate + 10000 * (y / rate)).abs() as u64;
				//~ let seed2 = (x / rate + 1 + 10000 * (y / rate)).abs() as u64;
				//~ let seed3 = (x / rate + 10000 * (y / rate + 1)).abs() as u64;
				//~ let seed4 = (x / rate + 1 + 10000 * (y / rate + 1)).abs() as u64;

				//~ let vertex1 = GridVertex::new(seed1);
				//~ let vertex2 = GridVertex::new(seed2);
				//~ let vertex3 = GridVertex::new(seed3);
				//~ let vertex4 = GridVertex::new(seed4);

				// This is dumb.
				let mut vertex = None;
				for cell in &self.cells
				{
					let point = Point3::new(shift_x, 0., shift_y);
					if cell.contains(&point)
					{
						vertex = cell.get_vertex(&point);
					}
				}
				let vertex = vertex.unwrap();

				//~ let vertex12 = vertex1.interpolate(&vertex2, fx);
				//~ let vertex34 = vertex3.interpolate(&vertex4, fx);
				//~ let vertex = vertex12.interpolate(&vertex34, fy);
				//~ let vertex = vertex1;

				for branch in &vertex.branches
				{
					for idx in 0..branch.len() - 1
					{
						for (x, y, z) in &branch[idx..idx + 2]
						{
							vertices.push(Vertex {
								x: x + shift_x,
								y: *y,
								z: z + shift_y,
								u: 0.,
								v: 0.,
								color: vertex.color,
							})
						}
					}
				}
			}
		}

		for (_, (position, drawable, stats)) in
			self.world.query::<(&Position, &Drawable, &Stats)>().iter()
		{
			let pos = position.pos;
			let dir = position.dir;

			for i in 0..40
			{
				for j in 0..2
				{
					let theta = 12. * std::f32::consts::PI * (i + j) as f32 / 40.;
					let dx = theta / 3.;
					let dz = stats.size * (1. - i as f32 / 40.) * theta.sin();
					let dy = stats.size * (1. - i as f32 / 40.) * theta.cos();

					let color = match drawable.kind
					{
						DrawKind::Player => Color::from_rgb_f(1., 1., i as f32 / 40.),
						DrawKind::Enemy => Color::from_rgb_f(1., 0., i as f32 / 40.),
						DrawKind::Bullet => Color::from_rgb_f(0.2, 0.2, 1.),
						DrawKind::Hit => Color::from_rgb_f(1., 1., 0.),
					};

					let vtx_pos = Point2::new(dx, dz);
					let rot = Rotation2::new(dir);
					let vtx_pos = rot * vtx_pos;

					vertices.push(Vertex {
						x: pos.x + vtx_pos.x,
						y: stats.size + dy,
						z: pos.z + vtx_pos.y,
						u: 0.,
						v: 0.,
						color: color,
					})
				}
			}
		}

		for (_, (position, blade_blade, stats)) in self
			.world
			.query::<(&Position, &BladeBlade, &Stats)>()
			.iter()
		{
			let pos = position.pos;

			let radii = [1., 0.5, 0.3, 0.7, 0.1, 0.6, 0.4, 0.9, 0.2, 0.8];
			let offsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.];
			let speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.];

			for blade in 0..blade_blade.num_blades
			{
				let r = stats.area_of_effect * radii[blade as usize];

				let theta = 2.
					* std::f32::consts::PI
					* (state.time() as f32 / (1. - 0.5 * speeds[blade as usize])
						+ offsets[blade as usize]);
				let theta = theta.rem_euclid(2. * std::f32::consts::PI);

				let color = Color::from_rgb_f(0.2, 1., 0.2);

				let pts = [(-5., -5.), (0., 5.), (5., -5.)];
				for i in 0..(pts.len() - 1)
				{
					for (dx, dz) in &pts[i..i + 2]
					{
						let vtx_pos = Point2::new(dx + r, *dz);
						let rot = Rotation2::new(theta);
						let vtx_pos = rot * vtx_pos;

						vertices.push(Vertex {
							x: pos.x + vtx_pos.x,
							y: pos.y,
							z: pos.z + vtx_pos.y,
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
						let theta2 = -0.25 * std::f32::consts::PI * (i + j) as f32 / 10.;
						let dx = r * (theta2 + theta).cos();
						let dz = r * (theta2 + theta).sin();
						let dy = pos.y;

						let color = Color::from_rgb_f(0.2, 1., 0.2);

						vertices.push(Vertex {
							x: pos.x + dx,
							y: dy,
							z: pos.z + dz,
							u: 0.,
							v: 0.,
							color: color,
						})
					}
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
