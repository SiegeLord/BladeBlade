use crate::error::Result;
use crate::game_state::{GameState, NextScreen};
use crate::spatial_grid::{self, SpatialGrid};
use crate::speech::get_speech;
use crate::utils::{
	camera_project, get_ground_from_screen, load_obj, mat4_to_transform, max, min,
	projection_transform, random_color, sigmoid, ColorExt, Vec2D, Vec3D, DT, PI,
};
use crate::{controls, ui};
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

static CELL_SIZE: i32 = 1024;
static CELL_RADIUS: i32 = 2;
static SUPER_GRID: i32 = 8;
static VERTEX_RADIUS: f32 = 64.;

#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum EnemyKind
{
	Normal,
	Magic,
	Rare,
}

fn make_rare_name(rng: &mut impl Rng) -> String
{
	let prefix = [
		"Snotty",
		"Corpulent",
		"Fart",
		"Puke",
		"Omega",
		"Poop-Stained",
		"Incompetent",
		"Recently Promoted",
		"Laxative",
		"Stained",
		"Several Wipes",
		"World-Ending",
	];
	let suffix = [
		"Gas",
		"Puke",
		"Fart",
		"Bubble",
		"the Flatulent",
		"of the Crusade",
		"Who Clogged",
		"Stain",
		"Bad Smell",
	];

	let prefix_idx = rng.gen_range(0..prefix.len());
	let suffix_idx = rng.gen_range(0..suffix.len());
	format!("{} {}", prefix[prefix_idx], suffix[suffix_idx])
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[repr(i32)]
enum RewardKind
{
	Life = 0,
	Mana,
	Speed,
	CastDelay,
	LifeRegen,
	ManaRegen,
	AreaOfEffect,
	ManaCost,
	SpellDamage,
	ExplodeOnDeath,
	Dash,
	SkillDuration,
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
		match *self
		{
			RewardKind::ExplodeOnDeath => 80,
			RewardKind::Dash => 40,
			_ =>
			{
				let max_value = 5. * 1.1_f32.powf(tier as f32);
				let min_value = 1. * 1.1_f32.powf(tier as f32 - 1.);
				let mut value = rng.gen_range(min_value as i32..max_value as i32);
				if rng.gen::<bool>()
				{
					value = -value;
				}
				value
			}
		}
	}

	fn from_idx(idx: i32) -> Option<Self>
	{
		match idx
		{
			0 => Some(RewardKind::Life),
			1 => Some(RewardKind::Mana),
			2 => Some(RewardKind::Speed),
			3 => Some(RewardKind::CastDelay),
			4 => Some(RewardKind::LifeRegen),
			5 => Some(RewardKind::ManaRegen),
			6 => Some(RewardKind::AreaOfEffect),
			7 => Some(RewardKind::ManaCost),
			8 => Some(RewardKind::SpellDamage),
			9 => Some(RewardKind::ExplodeOnDeath),
			10 => Some(RewardKind::Dash),
			11 => Some(RewardKind::SkillDuration),
			_ => None,
		}
	}

	fn weight(&self, _tier: i32) -> i32
	{
		match *self
		{
			RewardKind::Life => 30,
			RewardKind::Mana => 30,
			RewardKind::Speed => 10,
			RewardKind::CastDelay => 2,
			RewardKind::LifeRegen => 15,
			RewardKind::ManaRegen => 15,
			RewardKind::AreaOfEffect => 20,
			RewardKind::ManaCost => 10,
			RewardKind::SpellDamage => 30,
			RewardKind::ExplodeOnDeath => 1,
			RewardKind::Dash => 5,
			RewardKind::SkillDuration => 15,
			RewardKind::NumRewards => 0,
		}
	}

	fn description(&self, value: i32) -> String
	{
		let plus_sign = |v| if v > 0 { "+" } else { "" };

		match *self
		{
			RewardKind::Life => format!("Life {}{}", plus_sign(value), value),
			RewardKind::Mana => format!("Mana {}{}", plus_sign(value), value),
			RewardKind::Speed => format!("Move speed {}{}", plus_sign(value), value),
			RewardKind::CastDelay => format!("Cast delay {}{}", plus_sign(-value), -value),
			RewardKind::LifeRegen => format!("% life regen {}{}", plus_sign(value), value),
			RewardKind::ManaRegen => format!("% mana regen {}{}", plus_sign(value), value),
			RewardKind::AreaOfEffect => format!("Area of effect {}{}", plus_sign(value), value),
			RewardKind::ManaCost => format!("Mana cost {}{}", plus_sign(-value), -value),
			RewardKind::SpellDamage => format!("Spell damage {}{}", plus_sign(value), value),
			RewardKind::ExplodeOnDeath => format!("Enemies explode on death"),
			RewardKind::Dash => format!("You can dash"),
			RewardKind::SkillDuration => format!("Skill duration {}{}", plus_sign(value), value),
			_ => "ERROR".into(),
		}
	}

	fn color(&self, value: i32) -> Color
	{
		match *self
		{
			RewardKind::ExplodeOnDeath | RewardKind::Dash => Color::from_rgb_f(1., 0.8, 0.3),
			_ =>
			{
				if self.is_positive(value)
				{
					Color::from_rgb_f(0.8, 0.8, 1.)
				}
				else
				{
					Color::from_rgb_f(1., 0.4, 0.4)
				}
			}
		}
	}

	fn apply(&self, value: i32, stats: &mut Stats)
	{
		match *self
		{
			RewardKind::Life =>
			{
				stats.max_life += value as f32;
				stats.max_life = max(1., stats.max_life);
			}
			RewardKind::Mana =>
			{
				stats.max_mana += value as f32;
				stats.max_mana = max(1., stats.max_mana);
			}
			RewardKind::Speed =>
			{
				stats.speed += value as f32;
			}
			RewardKind::CastDelay =>
			{
				stats.cast_delay -= value as f32 / 100.;
				stats.cast_delay = max(0.05, stats.cast_delay);
			}
			RewardKind::LifeRegen =>
			{
				stats.life_regen += value as f32;
				stats.life_regen = max(0., stats.life_regen);
			}
			RewardKind::ManaRegen =>
			{
				stats.mana_regen += value as f32;
				stats.mana_regen = max(1., stats.mana_regen);
			}
			RewardKind::AreaOfEffect =>
			{
				// The AoE nerf.
				stats.area_of_effect = (stats.area_of_effect.powi(2) + value as f32).sqrt();
			}
			RewardKind::ManaCost =>
			{
				stats.mana_cost -= value as f32 / 10.;
				stats.mana_cost = max(0., stats.mana_cost);
			}
			RewardKind::SpellDamage =>
			{
				stats.spell_damage += value as f32 / 3.;
			}
			RewardKind::ExplodeOnDeath =>
			{
				stats.has_explodey = true;
			}
			RewardKind::Dash =>
			{
				stats.has_dash = true;
			}
			RewardKind::SkillDuration =>
			{
				stats.skill_duration += value as f32 / 100.;
				stats.skill_duration = max(0., stats.skill_duration);
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

	fn color(&self) -> Color
	{
		self.kind.color(self.value)
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
	fn new(center: Point2<i32>, seed: u64, enemies_left: &mut i32, world: &mut hecs::World)
		-> Self
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

			let level = max(1, -center.y);
			let spawn_prob = min(0.95, level as f32 / 60.);

			let mut mob_kind = EnemyKind::Normal;
			let mut boss_kind = EnemyKind::Normal;

			if rng.gen_range(0. ..1.) < 0.1
			{
				mob_kind = EnemyKind::Magic;
				boss_kind = EnemyKind::Magic;
			}
			else if rng.gen_range(0. ..1.) < 0.2
			{
				boss_kind = EnemyKind::Rare;
				if rng.gen_range(0. ..1.) < 0.1
				{
					mob_kind = EnemyKind::Magic;
				}
			}

			'exit: for y in -2..=2
			{
				for x in -2..=2
				{
					let is_center = x == 0 && y == 0;
					let kind = if is_center { boss_kind } else { mob_kind };
					let spacing = if mob_kind == EnemyKind::Normal
					{
						50.
					}
					else
					{
						75.
					};
					let exp_bonus = match kind
					{
						EnemyKind::Normal => 1.,
						EnemyKind::Magic => 3.,
						EnemyKind::Rare => 12.,
					};

					let name = match kind
					{
						EnemyKind::Normal => "Pathetic Monster".to_string(),
						EnemyKind::Magic => "Monster that Read the Manual".to_string(),
						EnemyKind::Rare => make_rare_name(&mut rng),
					};

					let stats = Stats::enemy_stats(level, kind, &mut rng);
					let life = stats.apply_effects().max_life;

					let x = dx + spacing * x as f32;
					let z = dy + spacing * y as f32;

					if Cell::world_to_cell(&Point3::new(x, 0., z)) != center
					{
						//~ println!("Out of bounds enemy!");
						continue;
					}

					if rng.gen_range(0. ..1.) < spawn_prob || is_center
					{
						if *enemies_left == 0
						{
							break 'exit;
						}
						*enemies_left -= 1;

						world.spawn((
							Enemy {
								time_to_deaggro: 0.,
								fire_delay: 1.,
								name: name,
								experience: (exp_bonus * 2000. * ((level + 1) as f32).powf(0.9))
									as i32,
								level: level,
							},
							Position {
								pos: Point3::new(x, 15., z),
								dir: rng.gen_range(0. ..2. * PI),
							},
							TimeToMove { time_to_move: 0. },
							Velocity {
								vel: Vector3::new(0., 0., 0.),
							},
							Drawable {
								kind: DrawKind::Enemy(kind),
							},
							Target { pos: None },
							Collision {
								kind: CollisionKind::Enemy,
							},
							stats,
							Weapon {
								time_to_fire: 0.,
								range: 320.,
							},
							Life { life: life },
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
	AfterImage,
	Enemy(EnemyKind),
	Bullet(f32),
	Hit,
	Explosion
	{
		start_time: f64,
	},
}

#[derive(Clone)]
pub struct Drawable
{
	kind: DrawKind,
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum EffectKind
{
	Dash,
	ProjMulti,
	ExtraFast,
	ExtraStrong,
}

impl EffectKind
{
	fn description(&self) -> &str
	{
		match *self
		{
			EffectKind::Dash => "Dashes",
			EffectKind::ProjMulti => "Multiple Projectiles",
			EffectKind::ExtraFast => "Extra Fast",
			EffectKind::ExtraStrong => "Extra Strong",
		}
	}
}

#[derive(Debug, Clone)]
pub struct Effect
{
	pub effect_over_time: f64,
	pub kind: EffectKind,
}

impl Effect
{
	pub fn apply(&self, stats: &mut Stats)
	{
		match self.kind
		{
			EffectKind::Dash =>
			{
				stats.speed *= 10.;
			}
			EffectKind::ProjMulti =>
			{
				stats.proj_multi *= 3;
			}
			EffectKind::ExtraFast =>
			{
				stats.speed *= 2.;
			}
			EffectKind::ExtraStrong =>
			{
				stats.spell_damage *= 4.;
			}
		}
	}

	pub fn is_dash(&self) -> bool
	{
		self.kind == EffectKind::Dash
	}
}

#[derive(Clone)]
pub struct Dash
{
	pub time_to_dash: f64,
	pub time_to_spawn_afterimage: f64,
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
	pub has_dash: bool,
	pub has_explodey: bool,
	pub proj_multi: i32,
	pub proj_speed: f32,

	pub effects: Vec<Effect>,
}

impl Stats
{
	fn apply_effects(&self) -> Stats
	{
		let mut new_stats = self.clone();
		for effect in &self.effects
		{
			effect.apply(&mut new_stats);
		}
		new_stats
	}

	fn is_dashing(&self) -> bool
	{
		for effect in &self.effects
		{
			if effect.is_dash()
			{
				return true;
			}
		}
		false
	}

	fn player_stats() -> Self
	{
		Self {
			speed: 300.,
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
			has_dash: false,
			has_explodey: false,
			proj_multi: 1,
			proj_speed: 1.,

			effects: vec![],
		}
	}

	fn enemy_stats(level: i32, kind: EnemyKind, rng: &mut impl Rng) -> Self
	{
		let mut f = 1.1_f32.powi(level - 1);
		let mut f2 = 1.01_f32.powi(level - 1);
		let mut size = min(30., 20. * f2);
		let mut max_life = 100. * f;
		let mut effects = vec![];

		match kind
		{
			EnemyKind::Normal =>
			{}
			EnemyKind::Magic =>
			{
				f *= 1.5;
				f2 *= 1.2;
				size *= 1.5;
				max_life *= 2.;

				if rng.gen_range(0. ..1.) < 0.2
				{
					effects.push(Effect {
						effect_over_time: -1.,
						kind: EffectKind::ProjMulti,
					});
				}
			}
			EnemyKind::Rare =>
			{
				f *= 2.;
				f2 *= 1.2;
				size *= 2.;
				max_life *= 4.;

				if rng.gen_range(0. ..1.) < 0.2
				{
					effects.push(Effect {
						effect_over_time: -1.,
						kind: EffectKind::ProjMulti,
					});
				}
				if rng.gen_range(0. ..1.) < 0.2
				{
					effects.push(Effect {
						effect_over_time: -1.,
						kind: EffectKind::ExtraFast,
					});
				}
				if rng.gen_range(0. ..1.) < 0.2
				{
					effects.push(Effect {
						effect_over_time: -1.,
						kind: EffectKind::ExtraStrong,
					});
				}
			}
		}

		Self {
			speed: 100. * f2,
			aggro_range: 400.,
			close_enough_range: 300. / f2,
			size: size,
			cast_delay: 0.5 / f2,
			skill_duration: 1.,
			area_of_effect: 100.,
			spell_damage: 1. * f,
			max_life: max_life,
			max_mana: 100.,
			life_regen: 0.,
			mana_regen: 10.,
			mana_cost: 10.,
			has_dash: false,
			has_explodey: false,
			proj_multi: 1,
			proj_speed: 200. * f2,

			effects: effects,
		}
	}

	fn bullet_stats(speed: f32, size: f32) -> Self
	{
		Self {
			speed: speed,
			aggro_range: 0.,
			close_enough_range: 0.,
			size: size,
			cast_delay: 0.,
			skill_duration: 0.,
			area_of_effect: 100.,
			spell_damage: 1.,
			max_life: 100.,
			max_mana: 100.,
			life_regen: 10.,
			mana_regen: 10.,
			mana_cost: 10.,
			has_dash: false,
			has_explodey: false,
			proj_multi: 1,
			proj_speed: 1.,

			effects: vec![],
		}
	}

	fn explosion_stats(aoe: f32) -> Self
	{
		Self {
			speed: 200.,
			aggro_range: 0.,
			close_enough_range: 0.,
			size: aoe,
			cast_delay: 0.,
			skill_duration: 0.,
			area_of_effect: 100.,
			spell_damage: 1.,
			max_life: 100.,
			max_mana: 100.,
			life_regen: 10.,
			mana_regen: 10.,
			mana_cost: 10.,
			has_dash: false,
			has_explodey: false,
			proj_multi: 1,
			proj_speed: 1.,

			effects: vec![],
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
			has_dash: false,
			has_explodey: false,
			proj_multi: 1,
			proj_speed: 1.,

			effects: vec![],
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
	pub level: i32,
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
	InMenu,
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
					x += 24. * rng.gen_range(-1.0..1.0);
					y += 4. * rng.gen_range(-1.0..1.0);
					z += 24. * rng.gen_range(-1.0..1.0);
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
	player_level: i32,
	player_kills: i32,
	player_blades: i32,
	project: Perspective3<f32>,
	mouse_pos: (i32, i32),

	cells: Vec<Cell>,
	ui_font: Font,

	state: State,
	selection_made: bool,
	reward_selection: i32,
	rewards: Vec<Vec<Reward>>,

	time_to_hide_intro: f64,
	time_to_hide_dash: f64,

	seed: u64,

	player_obj: Vec<[Point3<f32>; 2]>,
	monster_obj: Vec<[Point3<f32>; 2]>,
	bullet_obj: Vec<[Point3<f32>; 2]>,
	hit_obj: Vec<[Point3<f32>; 2]>,
	explosion_obj: Vec<[Point3<f32>; 2]>,
	blade_obj: Vec<[Point3<f32>; 2]>,

	subscreens: Vec<ui::SubScreen>,
}

impl Map
{
	pub fn new(state: &mut GameState, seed: u64) -> Result<Self>
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
			Dash {
				time_to_dash: 0.,
				time_to_spawn_afterimage: 0.,
			},
		));

		let mut cells = vec![];
		let mut enemies_left = 1000;
		for y in -CELL_RADIUS..=CELL_RADIUS
		{
			for x in -CELL_RADIUS..=CELL_RADIUS
			{
				cells.push(Cell::new(
					Point2::new(x, y),
					seed,
					&mut enemies_left,
					&mut world,
				));
			}
		}

		state.cache_bitmap("data/face.png")?;
		state.sfx.cache_sample("data/blade_blade.ogg")?;
		state.sfx.cache_sample("data/ui1.ogg")?;
		state.sfx.cache_sample("data/ui2.ogg")?;
		state.sfx.cache_sample("data/hit.ogg")?;
		state.sfx.cache_sample("data/blade_hit.ogg")?;
		state.sfx.cache_sample("data/death.ogg")?;
		state.sfx.cache_sample("data/death_monster1.ogg")?;
		state.sfx.cache_sample("data/death_monster2.ogg")?;
		state.sfx.cache_sample("data/laugh.ogg")?;
		state.sfx.cache_sample("data/weapon.ogg")?;
		state.sfx.cache_sample("data/dash.ogg")?;
		state.sfx.cache_sample("data/explosion.ogg")?;

		Ok(Self {
			world: world,
			player: player,
			player_pos: player_pos,
			project: projection_transform(state.display_width, state.display_height),
			mouse_pos: (0, 0),
			cells: cells,
			ui_font: state
				.ttf
				//~ .load_ttf_font("data/Energon.ttf", 16, Flag::zero())
				//~ .load_ttf_font("data/DejaVuSans.ttf", 20, Flag::zero())
				.load_ttf_font("data/AvQest.ttf", 24, Flag::zero())
				.map_err(|_| "Couldn't load 'data/AvQest.ttf'".to_string())?,
			state: State::Normal,
			selection_made: false,
			reward_selection: 0,
			rewards: vec![],
			seed: seed,
			time_to_hide_intro: state.time() + 5.,
			time_to_hide_dash: -1.,
			player_obj: load_obj("data/player.obj")?,
			monster_obj: load_obj("data/monster.obj")?,
			bullet_obj: load_obj("data/bullet.obj")?,
			hit_obj: load_obj("data/hit.obj")?,
			explosion_obj: load_obj("data/explosion.obj")?,
			blade_obj: load_obj("data/blade.obj")?,
			player_level: 1,
			player_kills: 0,
			player_blades: 0,
			subscreens: vec![],
		})
	}

	pub fn resize(&mut self, state: &GameState)
	{
		self.project = projection_transform(state.display_width, state.display_height);
		self.subscreens = self.subscreens.iter().map(|s| s.remake(state)).collect();
	}

	pub fn logic(&mut self, state: &mut GameState) -> Result<()>
	{
		if self.state == State::LevelUp
		{
			if !self.selection_made
			{
				let cx = state.display_width / 2.;
				let cy = state.display_height / 2.;

				let dtheta = 2. * PI / 6.;
				let mouse_theta =
					(self.mouse_pos.1 as f32 - cy).atan2(self.mouse_pos.0 as f32 - cx);
				self.reward_selection =
					(6 + ((mouse_theta + dtheta / 2.) / dtheta).floor() as i32) % 6;
			}
			return Ok(());
		}

		if self.state != State::Normal
		{
			return Ok(());
		}

		let cell_pos = Cell::world_to_cell(&self.player_pos);
		let radius = (CELL_RADIUS as f32 + 0.5) * CELL_SIZE as f32 + 320.;
		let center_x = cell_pos.x as f32 * CELL_SIZE as f32;
		let center_y = cell_pos.y as f32 * CELL_SIZE as f32;

		let mut collidable_grid = SpatialGrid::new(
			(2. * radius / 128.) as usize,
			(2. * radius / 128.) as usize,
			128.,
			128.,
		);

		let mut to_die = vec![];

		for (id, (position, _, stats)) in
			self.world.query::<(&Position, &Collision, &Stats)>().iter()
		{
			let stats = stats.apply_effects();
			let pos = Point2::new(position.pos.x - center_x, position.pos.z - center_y);
			let disp = Vector2::new(stats.size, stats.size);
			collidable_grid.push(spatial_grid::entry(pos - disp, pos + disp, id));
		}

		let want_move = state.controls.get_action_state(controls::Action::Move) > 0.5;
		let want_dash = state.controls.get_action_state(controls::Action::Dash) > 0.5;
		if want_move || want_dash
		{
			let (x, y) = self.mouse_pos;
			if let (Ok(stats), Ok(mut target)) = (
				self.world.get::<&Stats>(self.player),
				self.world.get::<&mut Target>(self.player),
			)
			{
				let fx = -1. + 2. * x as f32 / state.display_width;
				let fy = -1. + 2. * y as f32 / state.display_height;
				let camera = self.make_camera();

				let ground_pos = get_ground_from_screen(fx, -fy, self.project, camera);
				target.pos = Some(ground_pos);
			}

			if want_dash
			{
				if let (Ok(position), Ok(mut stats), Ok(mut dash), Ok(mut mana)) = (
					self.world.get::<&mut Position>(self.player),
					self.world.get::<&mut Stats>(self.player),
					self.world.get::<&mut Dash>(self.player),
					self.world.get::<&mut Mana>(self.player),
				)
				{
					if stats.has_dash
					{
						let cost = 0.5 * stats.mana_cost;
						if state.time() > dash.time_to_dash && mana.mana > cost
						{
							dash.time_to_dash = state.time() + 1.5 * stats.skill_duration as f64;
							let effect = Effect {
								kind: EffectKind::Dash,
								effect_over_time: state.time() + 0.1 * stats.skill_duration as f64,
							};
							stats.effects.push(effect);
							mana.mana -= cost;

							state.sfx.play_positional_sound(
								"data/dash.ogg",
								Vec2D::new(position.pos.x, position.pos.z),
								Vec2D::new(self.player_pos.x, self.player_pos.z),
							)?;
						}
					}
				}
			}
		}

		// position -> target
		for (_, (position, velocity, target, stats)) in self
			.world
			.query::<(&Position, &mut Velocity, &mut Target, &Stats)>()
			.iter()
		{
			let stats = stats.apply_effects();

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
			let pos = Point2::new(position.pos.x - center_x, position.pos.z - center_y);
			let disp = Vector2::new(stats.size, stats.size);

			let query = collidable_grid.query_rect(pos - disp, pos + disp, |_| true);
			let mut res: Vec<_> = query.iter().map(|v| v.inner.clone()).collect();
			res.sort();
			res.dedup();

			for other_id in res
			{
				if let Some((other_position, other_collision, other_stats)) = self
					.world
					.query_one::<(&Position, &Collision, &Stats)>(other_id)?
					.get()
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
		}

		// velocity application
		for (id, (position, velocity)) in self.world.query::<(&mut Position, &Velocity)>().iter()
		{
			if let Ok(time_to_move) = self.world.get::<&TimeToMove>(id)
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
		if let Ok(player_pos) = self.world.get::<&Position>(self.player)
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
		let mut num_enemies = 0;
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
			let stats = stats.apply_effects();

			num_enemies += 1;
			if let Some(pos) = target.pos
			{
				//~ println!("Fire {:?}", id);
				if state.time() > weapon.time_to_fire
				{
					weapon.time_to_fire =
						state.time() + enemy.fire_delay as f64 + stats.cast_delay as f64;
					time_to_move.time_to_move = state.time() + stats.cast_delay as f64;

					let dtheta = PI / 12.;
					for i in 0..stats.proj_multi
					{
						let disp = (pos - position.pos).xz();
						let rot = Rotation2::new((i - stats.proj_multi / 2) as f32 * dtheta);
						let disp = rot * disp;
						let target = Vector3::new(disp.x, pos.y, disp.y);
						new_bullets.push((
							position.pos,
							position.pos + target,
							stats.proj_speed,
							stats.spell_damage,
						));
					}

					state.sfx.play_positional_sound(
						"data/weapon.ogg",
						Vec2D::new(position.pos.x, position.pos.z),
						Vec2D::new(self.player_pos.x, self.player_pos.z),
					)?;
				}
			}
		}
		if state.tick % 20 == 0
		{
			//~ println!("Num enemies {}", num_enemies);
		}

		for (start, dest, speed, spell_damage) in new_bullets
		{
			let dir = dest - start;
			let size = spell_damage.powf(1. / 3.) * 10.;
			let stats = Stats::bullet_stats(speed, size);
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
					kind: DrawKind::Bullet(sigmoid((spell_damage - 10.) / 10.)),
				},
				stats,
				TimeToDie {
					time_to_die: state.time() + 5.,
				},
			));
			//~ println!("Fired: {:?}", id);
		}

		// Bullet to player collision
		let mut hits = vec![];
		if let (Ok(player_pos), Ok(mut life), Ok(player_stats)) = (
			self.world.get::<&Position>(self.player),
			self.world.get::<&mut Life>(self.player),
			self.world.get::<&Stats>(self.player),
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
					state.sfx.play_positional_sound(
						"data/hit.ogg",
						Vec2D::new(player_pos.pos.x, player_pos.pos.z),
						Vec2D::new(self.player_pos.x, self.player_pos.z),
					)?;
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
			self.world.get::<&mut BladeBlade>(self.player),
			self.world.get::<&Stats>(self.player),
			self.world.get::<&mut TimeToMove>(self.player),
			self.world.get::<&mut Target>(self.player),
			self.world.get::<&mut Mana>(self.player),
			self.world.get::<&mut Position>(self.player),
		)
		{
			if state.controls.get_action_state(controls::Action::Blade) > 0.5
				&& state.time() > blade_blade.time_to_fire
				&& mana.mana > stats.mana_cost
			{
				blade_blade.time_to_fire = state.time() + stats.cast_delay as f64;
				blade_blade.time_to_lose_blade = state.time() + stats.skill_duration as f64;
				blade_blade.num_blades = min(10, blade_blade.num_blades + 1);
				time_to_move.time_to_move = state.time() + stats.cast_delay as f64;
				target.pos = None;
				mana.mana -= stats.mana_cost;
				self.player_blades += 1;

				state.sfx.play_positional_sound(
					"data/blade_blade.ogg",
					Vec2D::new(position.pos.x, position.pos.z),
					Vec2D::new(self.player_pos.x, self.player_pos.z),
				)?;
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
						state.sfx.play_positional_sound(
							"data/blade_hit.ogg",
							Vec2D::new(enemy_position.pos.x, enemy_position.pos.z),
							Vec2D::new(self.player_pos.x, self.player_pos.z),
						)?;
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
		let mut rng = thread_rng();
		for (id, (position, life)) in self.world.query::<(&Position, &Life)>().iter()
		{
			if life.life < 0.
			{
				to_die.push(id);

				if id == self.player
				{
					state.sfx.play_positional_sound(
						"data/death.ogg",
						Vec2D::new(position.pos.x, position.pos.z),
						Vec2D::new(self.player_pos.x, self.player_pos.z),
					)?;
				}
				else
				{
					state.sfx.play_positional_sound(
						["data/death_monster1.ogg", "data/death_monster2.ogg"][rng.gen_range(0..2)],
						Vec2D::new(position.pos.x, position.pos.z),
						Vec2D::new(self.player_pos.x, self.player_pos.z),
					)?;
				}
			}
		}

		// Enemy on death effects
		let mut explosions = vec![];
		if let (Ok(stats), Ok(mut life), Ok(mut mana), Ok(mut experience)) = (
			self.world.get::<&Stats>(self.player),
			self.world.get::<&mut Life>(self.player),
			self.world.get::<&mut Mana>(self.player),
			self.world.get::<&mut Experience>(self.player),
		)
		{
			for (_, (enemy, position, enemy_stats, enemy_life)) in self
				.world
				.query::<(&Enemy, &Position, &Stats, &Life)>()
				.iter()
			{
				if enemy_life.life < 0.
				{
					let level_diff = (experience.level - enemy.level).abs();
					let f = if level_diff < 5
					{
						1.
					}
					else
					{
						0.3_f32.powi(level_diff - 4)
					};
					experience.experience += ((enemy.experience as f32) * f) as i32;
					self.player_kills += 1;
					if stats.has_explodey
					{
						explosions.push((
							position.pos,
							0.01 * stats.spell_damage * enemy_stats.max_life,
							stats.area_of_effect,
						));
					}
				}
			}

			if experience.experience >= level_to_experience(experience.level + 1)
			{
				experience.level += 1;
				self.player_level += 1;
				life.life = stats.max_life;
				mana.mana = stats.max_mana;
				self.state = State::LevelUp;
				self.selection_made = false;
				state.paused = true;

				self.rewards.clear();
				for _ in 0..6
				{
					let mut reward_vec = vec![];
					'restart: loop
					{
						reward_vec.clear();
						let mut total_value = 0;
						for _ in 0..2
						{
							let reward = Reward::new(experience.level, &reward_vec[..]);
							total_value += reward.value;
							if reward.kind == RewardKind::ExplodeOnDeath && stats.has_explodey
							{
								continue 'restart;
							}
							if reward.kind == RewardKind::Dash && stats.has_dash
							{
								continue 'restart;
							}
							reward_vec.push(reward);
						}

						let positive_thresh = (20. * 1.15_f32.powi(experience.level)) as i32;
						let negative_thresh = -(10. * 1.15_f32.powi(experience.level)) as i32;
						if total_value > positive_thresh || total_value < negative_thresh
						{
							continue 'restart;
						}
						break;
					}
					reward_vec.sort_by_key(|r| r.kind as i32);
					self.rewards.push(reward_vec);
				}

				state.sfx.play_sound("data/laugh.ogg")?;
			}
		}

		for (pos, damage, aoe) in explosions
		{
			self.world.spawn((
				Position { pos: pos, dir: 0. },
				Stats::explosion_stats(aoe),
				Drawable {
					kind: DrawKind::Explosion {
						start_time: state.time(),
					},
				},
				TimeToDie {
					time_to_die: state.time() + 0.3,
				},
			));

			state.sfx.play_positional_sound(
				"data/explosion.ogg",
				Vec2D::new(pos.x, pos.z),
				Vec2D::new(self.player_pos.x, self.player_pos.z),
			)?;

			for (_, (_, position, life, stats)) in self
				.world
				.query::<(&Enemy, &Position, &mut Life, &Stats)>()
				.iter()
			{
				let disp = position.pos - pos;
				if disp.norm() < aoe + stats.size
				{
					life.life -= damage;
				}
			}
		}

		// Effect expiration
		for (_, stats) in self.world.query::<&mut Stats>().iter()
		{
			stats
				.effects
				.retain(|e| e.effect_over_time < 0. || (state.time() < e.effect_over_time));
		}

		// Dash after-images
		let mut after_image = None;
		for (_, (position, dash, stats)) in
			self.world.query::<(&Position, &mut Dash, &Stats)>().iter()
		{
			if stats.is_dashing()
			{
				if state.time() > dash.time_to_spawn_afterimage
				{
					after_image = Some((position.clone(), stats.clone()));
					dash.time_to_spawn_afterimage = state.time() + 0.01;
				}
			}
		}

		if let Some((position, stats)) = after_image
		{
			self.world.spawn((
				position,
				stats,
				Drawable {
					kind: DrawKind::AfterImage,
				},
				TimeToDie {
					time_to_die: state.time() + 0.3,
				},
			));
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
		if let Ok(player_pos) = self.world.get::<&Position>(self.player)
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
							if self.world.get::<&&Enemy>(id).is_ok()
							{
								num_enemies -= 1;
							}
						}
					}
				//~ println!("Killed {}", cell.center);
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
						//~ println!("New cell {}", cell_center);
					}
				}
			}
		}

		let mut enemies_left = 1000 - num_enemies;

		new_cell_centers.shuffle(&mut rng);

		for cell_center in new_cell_centers
		{
			self.cells.push(Cell::new(
				cell_center,
				self.seed,
				&mut enemies_left,
				&mut self.world,
			));
		}

		to_die.sort();
		to_die.dedup();

		// Remove dead entities
		for id in to_die
		{
			self.world.despawn(id)?;
		}

		// Recenter things around the player.
		//~ let mut center = Point3::new(0., 0., 0.);
		//~ if let Ok(player_pos) = self.world.get::<&Position>(self.player)
		//~ {
		//~ center.x = player_pos.pos.x;
		//~ center.z = player_pos.pos.z;
		//~ }

		//~ for (_, position) in self.world.query::<&mut Position>().iter()
		//~ {
		//~ position.pos.x -= center.x;
		//~ position.pos.z -= center.z;
		//~ }
		//~ for (_, target) in self.world.query::<&mut Target>().iter()
		//~ {
		//~ target.pos.as_mut().map(|v|{

		//~ v.x -= center.x;
		//~ v.z -= center.z;
		//~ });
		//~ }

		// Camera pos
		if let Ok(player_pos) = self.world.get::<&Position>(self.player)
		{
			self.player_pos = player_pos.pos;
		}

		Ok(())
	}

	pub fn input(&mut self, event: &Event, state: &mut GameState) -> Result<Option<NextScreen>>
	{
		state.controls.decode_event(event);
		match self.state
		{
			State::Normal => match event
			{
				Event::MouseButtonDown { x, y, .. } =>
				{
					self.mouse_pos = (*x, *y);
				}
				Event::MouseAxes { x, y, .. } =>
				{
					self.mouse_pos = (*x, *y);
				}
				Event::KeyDown { keycode, .. } => match *keycode
				{
					KeyCode::Escape =>
					{
						self.subscreens
							.push(ui::SubScreen::InGameMenu(ui::InGameMenu::new(
								state.display_width,
								state.display_height,
							)));
						self.state = State::InMenu;
						state.paused = true;
						state.sfx.play_sound("data/ui2.ogg")?;
					}
					_ => (),
				},
				_ => (),
			},
			State::LevelUp =>
			{
				let mut apply_rewards = false;
				match event
				{
					Event::MouseButtonDown { x, y, .. } =>
					{
						self.mouse_pos = (*x, *y);
						self.selection_made = true;
						let cx = state.display_width / 2.;
						let cy = state.display_height / 2.;

						let dtheta = 2. * PI / 6.;
						let mouse_theta =
							(self.mouse_pos.1 as f32 - cy).atan2(self.mouse_pos.0 as f32 - cx);
						self.reward_selection =
							(6 + ((mouse_theta + dtheta / 2.) / dtheta).floor() as i32) % 6;
						state.sfx.play_sound("data/ui1.ogg")?;
					}
					Event::MouseAxes { x, y, .. } =>
					{
						self.mouse_pos = (*x, *y);
					}
					Event::KeyDown { keycode, .. } => match *keycode
					{
						KeyCode::Escape =>
						{
							self.selection_made = false;
							state.sfx.play_sound("data/ui2.ogg")?;
						}
						KeyCode::Y =>
						{
							if self.selection_made
							{
								apply_rewards = true;
								state.sfx.play_sound("data/ui1.ogg")?;
							}
						}
						KeyCode::N =>
						{
							if self.selection_made
							{
								self.selection_made = false;
								state.sfx.play_sound("data/ui2.ogg")?;
							}
						}
						_ => (),
					},
					_ => (),
				}
				if apply_rewards
				{
					let rewards = &self.rewards[self.reward_selection as usize];

					if let Ok(mut stats) = self.world.get::<&mut Stats>(self.player)
					{
						for reward in rewards
						{
							reward.apply(&mut stats);
						}
						//~ dbg!(&*stats);
						if stats.has_dash && self.time_to_hide_dash < 0.
						{
							self.time_to_hide_dash = state.time() + 5.;
						}
					}
					self.state = State::Normal;
					state.paused = false;
				}
			}
			State::InMenu =>
			{
				if let Some(action) = self
					.subscreens
					.last_mut()
					.and_then(|s| s.input(state, event))
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
						ui::Action::Back =>
						{
							self.subscreens.pop().unwrap();
						}
						ui::Action::Restart =>
						{
							state.paused = false;
							return Ok(Some(NextScreen::Game));
						}
						ui::Action::MainMenu =>
						{
							state.paused = false;
							return Ok(Some(NextScreen::Menu));
						}
						_ => (),
					}
				}
				if self.subscreens.is_empty()
				{
					self.state = State::Normal;
					state.paused = false;
				}
			}
		}
		Ok(None)
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
			let stats = stats.apply_effects();
			let pos = position.pos;
			let dir = position.dir;

			if pos.x < top_left.x - 250.
				|| pos.x > top_right.x + 250.
				|| pos.z < top_left.z - 250.
				|| pos.z > bottom_left.z + 250.
			{
				continue;
			}

			let size = match drawable.kind
			{
				DrawKind::Explosion { start_time } =>
				{
					(0.5 + 0.5 * (state.time() - start_time) as f32) * stats.size
				}
				_ => stats.size,
			};

			let color =
				match drawable.kind
				{
					DrawKind::Player => Color::from_rgb_f(1., 0.5, 1.),
					DrawKind::AfterImage => Color::from_rgb_f(0.8, 0.8, 0.8),
					DrawKind::Enemy(kind) => match kind
					{
						EnemyKind::Normal => Color::from_rgb_f(0., 1., 0.),
						EnemyKind::Magic => Color::from_rgb_f(0.3, 0.3, 1.),
						EnemyKind::Rare => Color::from_rgb_f(1., 1., 0.),
					},
					DrawKind::Bullet(f) => Color::from_rgb_f(0.2, 0.2, 1.)
						.interpolate(Color::from_rgb_f(1., 1., 1.), f),
					DrawKind::Hit => Color::from_rgb_f(1., 0.1, 0.1),
					DrawKind::Explosion { .. } => Color::from_rgb_f(1., 0.5, 0.),
				};

			let obj = match drawable.kind
			{
				DrawKind::Player => &self.player_obj,
				DrawKind::AfterImage => &self.player_obj,
				DrawKind::Enemy(_) => &self.monster_obj,
				DrawKind::Bullet(_) => &self.bullet_obj,
				DrawKind::Hit => &self.hit_obj,
				DrawKind::Explosion { .. } => &self.explosion_obj,
			};

			for line in obj
			{
				for vtx in line
				{
					let vtx = vtx * size;
					let vtx_pos = Point2::new(vtx.x, vtx.z);
					let rot = Rotation2::new(dir);
					let vtx_pos = rot * vtx_pos;

					vertices.push(Vertex {
						x: pos.x + vtx_pos.x,
						y: vtx.y,
						z: pos.z + vtx_pos.y,
						u: 0.,
						v: 0.,
						color: color,
					})
				}
			}

			//~ for i in 0..40
			//~ {
			//~ for j in 0..2
			//~ {

			//~ }
			//~ }
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

				for line in &self.blade_obj
				{
					for vtx in line
					{
						let vtx = vtx * 15.;
						let vtx_pos = Point2::new(vtx.x + r, vtx.z);
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
			Matrix4::new_orthographic(0., state.display_width, state.display_height, 0., -1., 1.);

		state
			.core
			.use_projection_transform(&mat4_to_transform(ortho_mat));
		state.core.use_transform(&Transform::identity());
		state.core.set_depth_test(None);

		let lh = self.ui_font.get_line_height() as f32;
		let cx = state.display_width / 2.;
		let cy = state.display_height / 2.;

		if state.time() < self.time_to_hide_intro
		{
			let c = 0.6 + 0.4 * 0.5 * ((5. * state.core.get_time()).sin() + 1.) as f32;
			let color = Color::from_rgb_f(c, c, c);
			state.core.draw_text(
				&self.ui_font,
				color,
				cx,
				cy + 50.,
				FontAlign::Centre,
				&format!(
					"Press {} to activate Blade Blade",
					state
						.options
						.controls
						.get_action_string(controls::Action::Blade)
				),
			);
		}

		if self.time_to_hide_dash > 0. && state.time() < self.time_to_hide_dash
		{
			let c = 0.6 + 0.4 * 0.5 * ((5. * state.core.get_time()).sin() + 1.) as f32;
			let color = Color::from_rgb_f(c, c, c);
			state.core.draw_text(
				&self.ui_font,
				color,
				cx,
				cy + 50.,
				FontAlign::Centre,
				&format!(
					"Press {} to Dash",
					state
						.options
						.controls
						.get_action_string(controls::Action::Dash)
				),
			);
		}

		// Life/Mana

		let r = 125.;
		let dx = r * 1.2;
		let dy = state.display_height - r * 1.2;

		let mut f = -1.;
		if let (Ok(life), Ok(stats)) = (
			self.world.get::<&Life>(self.player),
			self.world.get::<&Stats>(self.player),
		)
		{
			f = 2. * (life.life / stats.max_life) - 1.;

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				dx,
				dy - r * 1.2 - lh / 2.,
				FontAlign::Centre,
				&format!(
					"{} / {}",
					life.life.ceil() as i32,
					stats.max_life.ceil() as i32,
				),
			)
		}

		draw_orb(state, r, dx, dy, f, Color::from_rgb_f(0.7, 0.1, 0.1));

		let dx = state.display_width - r * 1.2;

		let mut f = -1.;
		if let (Ok(mana), Ok(stats)) = (
			self.world.get::<&Mana>(self.player),
			self.world.get::<&Stats>(self.player),
		)
		{
			f = 2. * (mana.mana / stats.max_mana) - 1.;

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				dx,
				dy - r * 1.2 - lh / 2.,
				FontAlign::Centre,
				&format!(
					"{} / {}",
					mana.mana.ceil() as i32,
					stats.max_mana.ceil() as i32,
				),
			)
		}

		// Experience

		if let Ok(experience) = self.world.get::<&Experience>(self.player)
		{
			let dx = state.display_width / 2.;
			let dy = state.display_height - 64.;

			let w = state.display_width - r * 5.;
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
		let fx = -1. + 2. * x as f32 / state.display_width;
		let fy = -1. + 2. * y as f32 / state.display_height;
		let ground_pos = get_ground_from_screen(fx, -fy, self.project, camera);

		// Enemy health bar

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
				let f = max(0., life.life / stats.max_life);

				let w = 400.;
				let h = 32.;
				let dx = state.display_width / 2. - w / 2.;
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

				let dy = dy - lh / 2. + h / 2.;

				state.core.draw_text(
					&self.ui_font,
					Color::from_rgb_f(1., 1., 1.),
					state.display_width / 2.,
					dy,
					FontAlign::Centre,
					&enemy.name,
				);

				state.core.draw_text(
					&self.ui_font,
					Color::from_rgb_f(1., 1., 1.),
					state.display_width / 2.,
					dy + lh * 1.2,
					FontAlign::Centre,
					&format!("Level: {}", enemy.level),
				);

				for (i, effect) in stats.effects.iter().enumerate()
				{
					state.core.draw_text(
						&self.ui_font,
						Color::from_rgb_f(1., 1., 1.),
						state.display_width / 2.,
						dy + lh * 1.2 + lh * 1.2 * (i + 1) as f32,
						FontAlign::Centre,
						effect.kind.description(),
					);
				}
			}
		}

		if self.state == State::Normal && self.world.entity(self.player).is_err()
		{
			state.prim.draw_filled_rectangle(
				0.,
				0.,
				state.display_width,
				state.display_height,
				Color::from_rgba_f(0., 0., 0., 0.5),
			);

			let mut dy = cy - lh / 2. - lh * 3.;
			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				cx,
				dy,
				FontAlign::Centre,
				"You died!",
			);
			dy += lh * 1.2;
			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				cx,
				dy,
				FontAlign::Centre,
				&format!("You reached level {}", self.player_level),
			);
			dy += lh * 1.2;
			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				cx,
				dy,
				FontAlign::Centre,
				&format!("You killed {} monsters", self.player_kills),
			);
			dy += lh * 1.2;
			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				cx,
				dy,
				FontAlign::Centre,
				&format!("You summoned {} blades", self.player_blades),
			);
			dy += lh * 1.2;
			dy += lh * 1.2;
			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(1., 1., 1.),
				cx,
				dy,
				FontAlign::Centre,
				"Your deeds of valor will be remembered",
			);
		}

		// Level up

		if self.state == State::LevelUp
		{
			state.prim.draw_filled_rectangle(
				0.,
				0.,
				state.display_width,
				state.display_height,
				Color::from_rgba_f(0., 0., 0., 0.7),
			);

			let bmp = state.get_bitmap("data/face.png").unwrap();
			let bw = bmp.get_width() as f32;
			let bh = bmp.get_height() as f32;
			let r = 300.;
			state.core.draw_tinted_bitmap(
				bmp,
				Color::from_rgb_f(0.8, 0.2, 0.2),
				cx - bw / 2.,
				cy - bh / 2.,
				Flag::zero(),
			);

			let mut level = 0;

			if let Ok(experience) = self.world.get::<&Experience>(self.player)
			{
				level = experience.level;
			}

			let mut text = &format!("You reached level {}!", level)[..];
			let mut color = Color::from_rgb_f(1., 1., 1.);
			if self.selection_made
			{
				let c = 0.6 + 0.4 * 0.5 * ((5. * state.core.get_time()).sin() + 1.) as f32;
				color = Color::from_rgb_f(c, c, c);
				text = "Confirm your selection: (Y)";
			}

			state.core.draw_text(
				&self.ui_font,
				color,
				cx,
				cy - 1.3 * r,
				FontAlign::Centre,
				text,
			);

			let dtheta = 2. * PI / 6.;

			for (i, rewards) in self.rewards.iter().enumerate()
			{
				let theta = dtheta * i as f32;

				let x = cx + theta.cos() * r;
				let y = cy + theta.sin() * r;

				for (j, reward) in rewards.iter().enumerate()
				{
					state.core.draw_text(
						&self.ui_font,
						reward.color().interpolate(
							Color::from_rgb_f(0., 0., 0.),
							if i as i32 == self.reward_selection
							{
								0.
							}
							else
							{
								0.5
							},
						),
						x,
						y + lh * j as f32 - lh,
						if theta.cos() > 0.
						{
							FontAlign::Left
						}
						else
						{
							FontAlign::Right
						},
						&reward.description(),
					);
				}
			}

			state.core.draw_text(
				&self.ui_font,
				Color::from_rgb_f(0.8, 0.2, 0.2),
				cx,
				cy + 1.3 * r,
				FontAlign::Centre,
				get_speech(level, self.seed),
			);
		}

		if let Some(subscreen) = self.subscreens.last()
		{
			state.prim.draw_filled_rectangle(
				0.,
				0.,
				state.display_width,
				state.display_height,
				Color::from_rgba_f(0., 0., 0., 0.7),
			);
			subscreen.draw(state);
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
