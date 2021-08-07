use crate::error::Result;
use crate::game_state::{GameState, NextScreen};
use crate::utils::{
	camera_project, get_ground_from_screen, mat4_to_transform, max, min, projection_transform,
	random_color, ColorExt, Vec3D, DT, PI,
};
use allegro::*;
use allegro_font::*;
use allegro_primitives::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point2, Point3, Quaternion, RealField, Rotation2, Rotation3,
	Unit, Vector2, Vector3, Vector4,
};
use nalgebra as na;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

static CELL_SIZE: i32 = 2048;
static CELL_RADIUS: i32 = 2;
static SUPER_GRID: i32 = 8;
static VERTEX_RADIUS: f32 = 64.;

#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[repr(i32)]
enum RewardKind
{
	Life = 0,
	Mana,
	Speed,
	NumRewards,
}

impl RewardKind
{
	fn is_positive(&self, value: i32) -> bool
	{
		value > 0
	}

	fn gen_value(&self, tier: i32, rng: &mut impl Rng) -> i32
	{
		if rng.gen::<bool>()
		{
			rng.gen_range(1..5 * (tier + 1))
		}
		else
		{
			rng.gen_range(-5 * (tier + 1)..0)
		}
	}

	fn from_idx(idx: i32) -> Option<Self>
	{
		match idx
		{
			0 => Some(RewardKind::Life),
			1 => Some(RewardKind::Mana),
			2 => Some(RewardKind::Speed),
			_ => None,
		}
	}

	fn weight(&self, _tier: i32) -> i32
	{
		match *self
		{
			RewardKind::Life | RewardKind::Mana | RewardKind::Speed => 1,
			_ => 0,
		}
	}

	fn description(&self, value: i32) -> String
	{
		let plus_sign = |v| if v > 0 { "+" } else { "" };

		match *self
		{
			RewardKind::Life => format!("Life {}{}", plus_sign(value), value),
			RewardKind::Mana => format!("Mana {}{}", plus_sign(value), value),
			RewardKind::Speed => format!("Speed {}{}", plus_sign(value), value),
			_ => "ERROR".into(),
		}
	}

	fn apply(&self, value: i32, stats: &mut Stats)
	{
		match *self
		{
			RewardKind::Life =>
			{
				stats.max_life += value as f32;
			}
			RewardKind::Mana =>
			{
				stats.max_mana += value as f32;
			}
			RewardKind::Speed =>
			{
				stats.speed += value as f32;
			}
			_ => unreachable!(),
		}
	}
}

struct Reward
{
	tier: i32,
	value: i32,
	kind: RewardKind,
}

impl Reward
{
	fn new(tier: i32, exclude_rewards: &[Reward]) -> Self
	{
		let mut rng = thread_rng();

		let mut weights = Vec::with_capacity(RewardKind::NumRewards as usize);
		for idx in 0..RewardKind::NumRewards as i32
		{
			let kind = RewardKind::from_idx(idx).unwrap();
			weights.push(kind.weight(tier));
		}

		let dist = WeightedIndex::new(&weights).unwrap();

		let mut kind;
		'restart: loop
		{
			kind = RewardKind::from_idx(dist.sample(&mut rng) as i32).unwrap();
			for other_kind in exclude_rewards
			{
				if other_kind.kind == kind
				{
					continue 'restart;
				}
			}
			break;
		}
		let value = kind.gen_value(tier, &mut rng);
		Self {
			kind: kind,
			tier: tier,
			value: value,
		}
	}

	fn description(&self) -> String
	{
		self.kind.description(self.value)
	}

	fn apply(&self, stats: &mut Stats)
	{
		self.kind.apply(self.value, stats);
	}
}

#[derive(Clone)]
pub struct Cell
{
	center: Point2<i32>,
	vertices: Vec<GridVertex>,
}

impl Cell
{
	fn new(center: Point2<i32>, seed: u64, world: &mut hecs::World) -> Self
	{
		let mut rng = thread_rng();
		let world_center =
			Point2::new((center.x * CELL_SIZE) as f32, (center.y * CELL_SIZE) as f32);

		let w = CELL_SIZE as f32 / 2. - 100.;

		for _ in 0..3
		{
			let mut dx = world_center.x + rng.gen_range(-w..w);
			let mut dy = world_center.y + rng.gen_range(-w..w);

			if center.x == 0 || center.y == 0
			{
				loop
				{
					dx = world_center.x + rng.gen_range(-w..w);
					dy = world_center.y + rng.gen_range(-w..w);

					if (dx * dx + dy * dy) > 400. * 400.
					{
						break;
					}
				}
			}

			for y in -1..=1
			{
				for x in -1..=1
				{
					world.spawn((
						Enemy {
							time_to_deaggro: 0.,
							fire_delay: 1.,
							name: "Basic Enemy".into(),
							experience: 1000,
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
						Life { life: 100. },
					));
				}
			}
		}

		let num_vertices = (CELL_SIZE as f32 / VERTEX_RADIUS) as i32 / SUPER_GRID;
		let mut vertices = vec![];

		for y in -num_vertices..num_vertices + 1
		{
			for x in -num_vertices..num_vertices + 1
			{
				let grid_seed = (x + center.x * num_vertices * 2)
					+ (num_vertices * num_vertices) * (y + center.y * num_vertices * 2);
				vertices.push(GridVertex::new(grid_seed as u64 + seed));
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

#[derive(Debug, Clone)]
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
	pub max_life: f32,
	pub max_mana: f32,
	pub life_regen: f32,
	pub mana_regen: f32,
	pub mana_cost: f32,
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
			max_life: 100.,
			max_mana: 100.,
			life_regen: 1.,
			mana_regen: 5.,
			mana_cost: 10.,
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
			max_life: 100.,
			max_mana: 100.,
			life_regen: 0.,
			mana_regen: 10.,
			mana_cost: 10.,
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
			max_life: 100.,
			max_mana: 100.,
			life_regen: 10.,
			mana_regen: 10.,
			mana_cost: 10.,
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
			max_life: 100.,
			max_mana: 100.,
			life_regen: 10.,
			mana_regen: 10.,
			mana_cost: 10.,
		}
	}
}

#[derive(Clone)]
pub struct Enemy
{
	pub time_to_deaggro: f64,
	pub fire_delay: f32,
	pub name: String,
	pub experience: i32,
}

#[derive(Clone)]
pub struct Experience
{
	level: i32,
	experience: i32,
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
pub struct Life
{
	life: f32,
}

#[derive(Clone)]
pub struct Mana
{
	mana: f32,
}

#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
enum State
{
	Normal,
	LevelUp,
	Restart,
	Quit,
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
			let theta = 2. * PI * i as f32 / 3.;
			for j in 0..5
			{
				let mut x = j as f32 * VERTEX_RADIUS * theta.cos() / 4.;
				let mut z = j as f32 * VERTEX_RADIUS * theta.sin() / 4.;
				let mut y = 0.;
				if j != 0 && j != 4
				{
					x += 12. * rng.gen_range(-1.0..1.0);
					y += 4. * rng.gen_range(-1.0..1.0);
					z += 12. * rng.gen_range(-1.0..1.0);
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

fn level_to_experience(level: i32) -> i32
{
	1000 * level.pow(3)
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
	ui_font: Font,

	state: State,
	old_state: State,
	old_paused: bool,
	reward_selection: i32,
	rewards: Vec<Vec<Reward>>,

	seed: u64,
}

impl Map
{
	pub fn new(
		state: &GameState, seed: u64, display_width: f32, display_height: f32,
	) -> Result<Self>
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
			Life { life: 100. },
			Mana { mana: 100. },
			Experience {
				level: 1,
				experience: level_to_experience(1),
			},
		));

		let mut cells = vec![];
		for y in -CELL_RADIUS..=CELL_RADIUS
		{
			for x in -CELL_RADIUS..=CELL_RADIUS
			{
				cells.push(Cell::new(Point2::new(x, y), seed, &mut world));
			}
		}

		Ok(Self {
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
			ui_font: state
				.ttf
				.load_ttf_font("data/Energon.ttf", 16, Flag::zero())
				.map_err(|_| "Couldn't load 'data/Energon.ttf'".to_string())?,
			state: State::Normal,
			old_state: State::Normal,
			old_paused: false,
			reward_selection: 0,
			rewards: vec![],
			seed: seed,
		})
	}

	pub fn logic(&mut self, state: &mut GameState) -> Result<()>
	{
		if self.state == State::LevelUp
		{
			let cx = self.display_width / 2.;
			let cy = self.display_height / 2.;

			let dtheta = 2. * PI / 6.;
			let mouse_theta = (self.mouse_pos.1 as f32 - cy).atan2(self.mouse_pos.0 as f32 - cx);
			self.reward_selection = (6 + ((mouse_theta + dtheta / 2.) / dtheta).floor() as i32) % 6;

			return Ok(());
		}

		if self.state != State::Normal
		{
			return Ok(());
		}

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
			for (_id, (enemy, position, target, weapon, stats)) in self
				.world
				.query::<(&mut Enemy, &mut Position, &mut Target, &mut Weapon, &Stats)>()
				.iter()
			{
				let pos = position.pos;
				let dist = (player_pos.pos - pos).norm();
				if dist < stats.aggro_range || state.time() < enemy.time_to_deaggro
				{
					//~ println!("Aggro {:?}", id);
					if target.pos.is_none()
					{
						weapon.time_to_fire = state.time() + enemy.fire_delay as f64;
					}
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
		if let (Ok(player_pos), Ok(mut life), Ok(player_stats)) = (
			self.world.get::<Position>(self.player),
			self.world.get_mut::<Life>(self.player),
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
					life.life -= bullet.damage;
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
			Ok(mut mana),
			Ok(position),
		) = (
			self.world.get_mut::<BladeBlade>(self.player),
			self.world.get::<Stats>(self.player),
			self.world.get_mut::<TimeToMove>(self.player),
			self.world.get_mut::<Target>(self.player),
			self.world.get_mut::<Mana>(self.player),
			self.world.get_mut::<Position>(self.player),
		)
		{
			if self.space_state
				&& state.time() > blade_blade.time_to_fire
				&& mana.mana > stats.mana_cost
			{
				blade_blade.time_to_fire = state.time() + stats.cast_delay as f64;
				blade_blade.time_to_lose_blade = state.time() + stats.skill_duration as f64;
				blade_blade.num_blades = min(10, blade_blade.num_blades + 1);
				time_to_move.time_to_move = state.time() + stats.cast_delay as f64;
				target.pos = None;
				mana.mana -= stats.mana_cost;
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

				for (_id, (_, enemy_position, enemy_stats, mut life)) in self
					.world
					.query::<(&mut Enemy, &mut Position, &Stats, &mut Life)>()
					.iter()
				{
					let dist = (position.pos - enemy_position.pos).norm();
					if dist < stats.area_of_effect + enemy_stats.size
					{
						hits.push(enemy_position.pos);
						life.life -= stats.spell_damage;
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

		// Life
		for (id, life) in self.world.query::<&Life>().iter()
		{
			if life.life < 0.
			{
				to_die.push(id);
			}
		}

		// Experience on death
		if let Ok(mut experience) = self.world.get_mut::<Experience>(self.player)
		{
			for (_, (enemy, life)) in self.world.query::<(&Enemy, &Life)>().iter()
			{
				if life.life < 0.
				{
					experience.experience += enemy.experience;
				}
			}

			if experience.experience >= level_to_experience(experience.level + 1)
			{
				experience.level += 1;
				self.state = State::LevelUp;
				state.paused = true;

				self.rewards.clear();
				for _ in 0..6
				{
					let mut reward_vec = vec![];
					for _ in 0..2
					{
						reward_vec.push(Reward::new(0, &reward_vec[..]))
					}
					reward_vec.sort_by_key(|r| r.kind as i32);
					self.rewards.push(reward_vec);
				}
			}
		}

		// Life/mana regen
		for (_, (life, stats)) in self.world.query::<(&mut Life, &Stats)>().iter()
		{
			life.life += (stats.max_life * stats.life_regen / 100.) * DT;
			if life.life > stats.max_life
			{
				life.life = stats.max_life;
			}
		}

		for (_, (mana, stats)) in self.world.query::<(&mut Mana, &Stats)>().iter()
		{
			mana.mana += (stats.max_mana * stats.mana_regen / 100.) * DT;
			if mana.mana > stats.max_mana
			{
				mana.mana = stats.max_mana;
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
			self.cells
				.push(Cell::new(cell_center, self.seed, &mut self.world));
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

	pub fn input(&mut self, event: &Event, state: &mut GameState) -> Result<Option<NextScreen>>
	{
		let mut ret = None;
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

				if self.state == State::LevelUp
				{
					let rewards = &self.rewards[self.reward_selection as usize];

					if let Ok(mut stats) = self.world.get_mut::<Stats>(self.player)
					{
						for reward in rewards
						{
							reward.apply(&mut stats);
						}
					}
					self.state = State::Normal;
					state.paused = false;
				}
			}
			Event::MouseAxes { x, y, .. } =>
			{
				self.mouse_pos = (*x, *y);
			}
			Event::KeyDown { keycode, .. } => match *keycode
			{
				KeyCode::Space => self.space_state = true,
				KeyCode::P =>
				{
					state.paused = !state.paused;
					if self.state == State::Normal
					{
						self.state = State::LevelUp;
					}
					else
					{
						self.state = State::Normal;
					}
				}
				KeyCode::R =>
				{
					if self.state == State::Normal
					{
						state.paused = true;
						self.old_state = self.state;
						self.state = State::Restart
					}
				}
				KeyCode::Escape =>
				{
					if self.state == State::Quit
					{
						self.state = self.old_state;
						state.paused = self.old_paused;
					}
					else if self.state == State::Restart
					{
						self.state = self.old_state;
						state.paused = self.old_paused;
					}
					else if self.state == State::Normal
					{
						self.old_paused = state.paused;
						self.old_state = self.state;

						state.paused = true;
						self.state = State::Quit;
					}
				}
				KeyCode::Y =>
				{
					if self.state == State::Quit
					{
						ret = Some(NextScreen::Quit);
						state.paused = false;
					}
					else if self.state == State::Restart
					{
						ret = Some(NextScreen::Game);
						state.paused = false;
					}
				}
				KeyCode::N =>
				{
					if self.state == State::Quit
					{
						self.state = self.old_state;
						state.paused = self.old_paused;
					}
					else if self.state == State::Restart
					{
						self.state = self.old_state;
						state.paused = self.old_paused;
					}
				}
				_ => (),
			},
			Event::KeyUp { keycode, .. } =>
			{
				if *keycode == KeyCode::Space
				{
					self.space_state = false;
				}
			}
			_ => (),
		}

		Ok(ret)
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

	pub fn draw(&self, state: &GameState) -> Result<()>
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

		let theta = PI / 3.;
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

			if pos.x < top_left.x - 250.
				|| pos.x > top_right.x + 250.
				|| pos.z < top_left.z - 250.
				|| pos.z > bottom_left.z + 250.
			{
				continue;
			}

			for i in 0..40
			{
				for j in 0..2
				{
					let theta = 12. * PI * (i + j) as f32 / 40.;
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
			let speeds = [0.1, 0.3, 0.5, 0.7, 1.1, 1.3, 1.7, 1.9, 2.3, 3.1];

			for blade in 0..blade_blade.num_blades
			{
				let r = stats.area_of_effect * radii[blade as usize];

				let theta = 2.
					* PI * (state.time() as f32 / (1. - 0.5 * speeds[blade as usize] / 3.)
					+ offsets[blade as usize]);
				let theta = theta.rem_euclid(2. * PI);

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
						let theta2 = -0.25 * PI * (i + j) as f32 / 10.;
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

		// UI

		let ortho_mat =
			Matrix4::new_orthographic(0., self.display_width, self.display_height, 0., -1., 1.);

		state
			.core
			.use_projection_transform(&mat4_to_transform(ortho_mat));
		state.core.use_transform(&Transform::identity());
		state.core.set_depth_test(None);

		// Life/Mana

		let r = 125.;
		let dx = r * 1.2;
		let dy = self.display_height - r * 1.2;

		let mut f = -1.;
		if let (Ok(life), Ok(stats)) = (
			self.world.get::<Life>(self.player),
			self.world.get::<Stats>(self.player),
		)
		{
			f = 2. * (life.life / stats.max_life) - 1.;

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				dx,
				dy - r * 1.2,
				FontAlign::Centre,
				&format!(
					"{} / {}",
					life.life.ceil() as i32,
					stats.max_life.ceil() as i32,
				),
			)
		}

		draw_orb(state, r, dx, dy, f, Color::from_rgb_f(0.7, 0.1, 0.1));

		let dx = self.display_width - r * 1.2;

		let mut f = -1.;
		if let (Ok(mana), Ok(stats)) = (
			self.world.get::<Mana>(self.player),
			self.world.get::<Stats>(self.player),
		)
		{
			f = 2. * (mana.mana / stats.max_mana) - 1.;

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				dx,
				dy - r * 1.2,
				FontAlign::Centre,
				&format!(
					"{} / {}",
					mana.mana.ceil() as i32,
					stats.max_mana.ceil() as i32,
				),
			)
		}

		// Experience

		if let Ok(experience) = self.world.get::<Experience>(self.player)
		{
			let dx = self.display_width / 2.;
			let dy = self.display_height - 64.;

			let w = self.display_width - r * 5.;
			let h = 16.;

			let old_breakpoint = level_to_experience(experience.level) as f32;
			let new_breakpoint = level_to_experience(experience.level + 1) as f32;

			let f =
				(experience.experience as f32 - old_breakpoint) / (new_breakpoint - old_breakpoint);

			state.prim.draw_filled_rectangle(
				dx - w / 2.,
				dy,
				dx - w / 2. + w * f,
				dy + h,
				Color::from_rgb_f(0.9, 0.9, 0.5),
			);
			state.prim.draw_rectangle(
				dx - w / 2.,
				dy,
				dx + w / 2.,
				dy + h,
				Color::from_rgb_f(0.2, 0.2, 0.2),
				2.,
			);

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				dx,
				dy - 2. * h,
				FontAlign::Centre,
				&format!("Level: {}", experience.level,),
			)
		}

		draw_orb(state, r, dx, dy, f, Color::from_rgb_f(0.1, 0.1, 0.7));

		let (x, y) = self.mouse_pos;
		let fx = -1. + 2. * x as f32 / self.display_width;
		let fy = -1. + 2. * y as f32 / self.display_height;
		let ground_pos = get_ground_from_screen(fx, -fy, self.project, camera);

		let mut best_id = None;
		let mut best_dist = 1000.;
		for (id, (_, _, position, stats)) in self
			.world
			.query::<(&Life, &Enemy, &Position, &Stats)>()
			.iter()
		{
			let dist = (ground_pos - position.pos).norm();
			if dist < 3. * stats.size
			{
				if dist < best_dist
				{
					best_dist = dist;
					best_id = Some(id);
				}
			}
		}

		if let Some(id) = best_id
		{
			if let Some((life, enemy, stats)) =
				self.world.query_one::<(&Life, &Enemy, &Stats)>(id)?.get()
			{
				let f = life.life / stats.max_life;

				let w = 400.;
				let h = 32.;
				let dx = self.display_width / 2. - w / 2.;
				let dy = 32.;

				state.prim.draw_rectangle(
					dx,
					dy,
					dx + w,
					dy + h,
					Color::from_rgb_f(0.2, 0.2, 0.2),
					2.,
				);
				state.prim.draw_filled_rectangle(
					dx,
					dy,
					dx + w * f,
					dy + h,
					Color::from_rgb_f(0.7, 0.1, 0.1),
				);

				state.core.draw_text(
					&self.ui_font,
					Color::from_rgb_f(1., 1., 1.),
					self.display_width / 2.,
					dy - self.ui_font.get_line_height() as f32 / 2. + h / 2.,
					FontAlign::Centre,
					&enemy.name,
				);
			}
		}

		let cx = self.display_width / 2.;
		let cy = self.display_height / 2.;

		if self.state == State::Normal && self.world.entity(self.player).is_err()
		{
			state.prim.draw_filled_rectangle(
				0.,
				0.,
				self.display_width,
				self.display_height,
				Color::from_rgba_f(0., 0., 0., 0.5),
			);

			let lh = self.ui_font.get_line_height() as f32;

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				cx,
				cy - lh / 2.,
				FontAlign::Centre,
				"Your deeds of valor will be remembered (Esc/R)",
			);
		}

		if self.state == State::LevelUp
		{
			state.prim.draw_filled_rectangle(
				0.,
				0.,
				self.display_width,
				self.display_height,
				Color::from_rgba_f(0., 0., 0., 0.5),
			);

			let dtheta = 2. * PI / 6.;

			for (i, rewards) in self.rewards.iter().enumerate()
			{
				let theta = dtheta * i as f32;

				let x = cx + theta.cos() * 300.;
				let y = cy + theta.sin() * 300.;
				let lh = self.ui_font.get_line_height() as f32;

				for (j, reward) in rewards.iter().enumerate()
				{
					state.core.draw_text(
						&self.ui_font,
						if i as i32 == self.reward_selection
						{
							Color::from_rgb_f(1., 1., 1.)
						}
						else
						{
							Color::from_rgb_f(0.7, 0.7, 0.7)
						},
						x,
						y + lh * j as f32,
						FontAlign::Centre,
						&reward.description(),
					);
				}
			}
		}
		else if self.state == State::Quit
		{
			state.prim.draw_filled_rectangle(
				0.,
				0.,
				self.display_width,
				self.display_height,
				Color::from_rgba_f(0., 0., 0., 0.5),
			);

			let lh = self.ui_font.get_line_height() as f32;

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				cx,
				cy - lh / 2.,
				FontAlign::Centre,
				"Quit? (Y/N)",
			);
		}
		else if self.state == State::Restart
		{
			state.prim.draw_filled_rectangle(
				0.,
				0.,
				self.display_width,
				self.display_height,
				Color::from_rgba_f(0., 0., 0., 0.5),
			);

			let lh = self.ui_font.get_line_height() as f32;

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				cx,
				cy - lh / 2.,
				FontAlign::Centre,
				"Restart? (Y/N)",
			);
		}

		Ok(())
	}
}

fn draw_orb(state: &GameState, r: f32, dx: f32, dy: f32, f: f32, color: Color)
{
	let dtheta = 2. * PI / 32.;

	let mut vertices = vec![];
	let theta_start = f.acos();
	let num_vertices = ((2. * PI - 2. * theta_start) / dtheta) as i32;
	let dtheta = (2. * PI - 2. * theta_start) / num_vertices as f32;
	for i in 0..=num_vertices
	{
		let theta = theta_start + dtheta * i as f32;
		vertices.push(Vertex {
			x: dx - r * theta.sin(),
			y: dy - r * theta.cos(),
			z: 0.,
			u: 0.,
			v: 0.,
			color: color,
		})
	}

	state.prim.draw_prim(
		&vertices[..],
		Option::<&Bitmap>::None,
		0,
		vertices.len() as u32,
		PrimType::TriangleFan,
	);

	state
		.prim
		.draw_circle(dx, dy, r, Color::from_rgb_f(0.2, 0.2, 0.2), 2.);
}
