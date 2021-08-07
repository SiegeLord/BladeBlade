use crate::error::{Error, Result};
use allegro::*;
use allegro_audio::*;
use allegro_color::*;
use nalgebra;
use rand::prelude::*;
use serde::de::DeserializeOwned;
use serde::ser::Serialize;
use slr_config::{from_element, to_element, ConfigElement, Source};
use std::{fs, path};

pub const DT: f32 = 1. / 120.;
pub const TILE: f32 = 64.;
pub const PI: f32 = std::f32::consts::PI;
pub type Vec2D = nalgebra::Vector2<f32>;
pub type Vec3D = nalgebra::Vector3<f32>;

use na::{
	Isometry3, Matrix4, Perspective3, Point3, Quaternion, RealField, Rotation3, Unit, Vector3,
	Vector4,
};
use nalgebra as na;

pub fn projection_transform(dw: f32, dh: f32) -> Perspective3<f32>
{
	Perspective3::new(dw / dh, f32::pi() / 2., 0.1, 2000.)
}

pub fn mat4_to_transform(mat: Matrix4<f32>) -> Transform
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

pub fn camera_project(x: f32, y: f32, z: f32, player_z: f32) -> Isometry3<f32>
{
	let eye = Point3::new(x, y, z);
	let target = Point3::new(x, 0., player_z);
	let view = Isometry3::look_at_rh(&eye, &target, &Vector3::y());
	view
}

pub fn random_color(seed: u64, saturation: f32, value: f32) -> Color
{
	let mut rng = StdRng::seed_from_u64(seed);
	Color::from_hsv(rng.gen_range(0. ..360.), saturation, value)
}

pub trait ColorExt
{
	fn interpolate(&self, other: Color, f: f32) -> Color;
}

impl ColorExt for Color
{
	fn interpolate(&self, other: Color, f: f32) -> Color
	{
		let fi = 1. - f;
		let (r, g, b, a) = self.to_rgba_f();
		let (or, og, ob, oa) = other.to_rgba_f();
		Color::from_rgba_f(
			r * fi + or * f,
			g * fi + og * f,
			b * fi + ob * f,
			a * fi + oa * f,
		)
	}
}

/// x/y need to be in [-1, 1]
pub fn get_ground_from_screen(
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

	let dir = end - start;
	let f = (-start.y) / dir.y;
	start + f * dir
}

pub fn max<T: PartialOrd>(x: T, y: T) -> T
{
	if x > y
	{
		x
	}
	else
	{
		y
	}
}

pub fn min<T: PartialOrd>(x: T, y: T) -> T
{
	if x < y
	{
		x
	}
	else
	{
		y
	}
}

pub fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T
{
	if x < min
	{
		min
	}
	else if x > max
	{
		max
	}
	else
	{
		x
	}
}

pub fn map_to_idx(map_pos: MapPos, size: usize) -> Option<usize>
{
	if map_pos.x >= 0 && map_pos.y >= 0 && map_pos.x < size as i32 && map_pos.y < size as i32
	{
		Some(map_pos.x as usize + map_pos.y as usize * size)
	}
	else
	{
		None
	}
}

pub fn idx_to_map(idx: usize, size: usize) -> MapPos
{
	let x = idx % size;
	let y = idx / size;
	MapPos::new(x as i32, y as i32)
}

pub fn get_octant(dx: i32, dy: i32) -> Option<i32>
{
	if dx != 0 || dy != 0
	{
		let theta = (dy as f32).atan2(dx as f32);
		let octant = (4. * theta / std::f32::consts::PI + 3.) as i32;
		Some((octant + 1) % 8)
	}
	else
	{
		None
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MapPos
{
	pub x: i32,
	pub y: i32,
}

impl MapPos
{
	pub fn new(x: i32, y: i32) -> MapPos
	{
		MapPos { x: x, y: y }
	}

	pub fn to_pos(&self) -> Vec2D
	{
		Vec2D::new(self.x as f32 * TILE, self.y as f32 * TILE)
	}

	pub fn center(&self) -> Vec2D
	{
		self.to_pos() + Vec2D::new(TILE / 2., TILE / 2.)
	}

	pub fn from_pos(pos: Vec2D) -> MapPos
	{
		MapPos::new((pos.x / TILE) as i32, (pos.y / TILE) as i32)
	}

	pub fn l2_dist_to(&self, other: MapPos) -> f32
	{
		let dx = self.x - other.x;
		let dy = self.y - other.y;
		((dx * dx + dy * dy) as f32).sqrt()
	}

	pub fn l1_dist_to(&self, other: MapPos) -> i32
	{
		let dx = self.x - other.x;
		let dy = self.y - other.y;
		dx.abs() + dy.abs()
	}

	pub fn linf_dist_to(&self, other: MapPos) -> i32
	{
		let dx = self.x - other.x;
		let dy = self.y - other.y;
		max(dx.abs(), dy.abs())
	}
}

pub fn world_to_screen(vec: Vec2D) -> Vec2D
{
	let rot = nalgebra::Rotation2::<f32>::new(std::f32::consts::PI / 12.);
	let vec = (rot * vec).component_mul(&Vec2D::new(1., (std::f32::consts::PI / 4.).sin()));
	vec
}

pub fn screen_to_world(vec: Vec2D) -> Vec2D
{
	let vec = vec.component_div(&Vec2D::new(1., (std::f32::consts::PI / 4.).sin()));
	let rot = nalgebra::Rotation2::<f32>::new(-std::f32::consts::PI / 12.);
	rot * vec
}

pub fn read_to_string(path: &str) -> Result<String>
{
	fs::read_to_string(path)
		.map_err(|e| Error::new(format!("Couldn't read '{}'", path), Some(Box::new(e))))
}

pub fn load_config<T: DeserializeOwned + Clone>(file: &str) -> Result<T>
{
	let contents = read_to_string(file)?;
	let mut source = Source::new(path::Path::new(file), &contents);
	let element = ConfigElement::from_source(&mut source)
		.map_err(|e| Error::new(format!("Config parsing error"), Some(Box::new(e))))?;
	from_element::<T>(&element, Some(&source))
		.map_err(|e| Error::new(format!("Config parsing error"), Some(Box::new(e))))
}

pub fn load_bitmap(core: &Core, file: &str) -> Result<Bitmap>
{
	Ok(Bitmap::load(&core, file).map_err(|_| format!("Couldn't load {}", file))?)
}

pub fn load_sample(audio: &AudioAddon, path: &str) -> Result<Sample>
{
	Ok(Sample::load(audio, path).map_err(|_| format!("Couldn't load '{}'", path))?)
}

#[test]
fn test_invertible()
{
	let world = Vec2D::new(3., 2.);
	let screen = world_to_screen(world);
	let world2 = screen_to_world(screen);
	assert!((world.x - world2.x).abs() < 1e-3);
	assert!((world.y - world2.y).abs() < 1e-3);

	let screen = Vec2D::new(3., 2.);
	let world = screen_to_world(screen);
	let screen2 = world_to_screen(world);
	assert!((screen.x - screen2.x).abs() < 1e-3);
	assert!((screen.y - screen2.y).abs() < 1e-3);
}
