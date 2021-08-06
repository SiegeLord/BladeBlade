use crate::error::{Error, Result};
use allegro::*;
use allegro_audio::*;
use nalgebra;
use serde::de::DeserializeOwned;
use serde::ser::Serialize;
use slr_config::{from_element, to_element, ConfigElement, Source};
use std::fs;
use std::path;

pub const DT: f32 = 1. / 120.;
pub const TILE: f32 = 64.;
pub const GRAVITY: f32 = 100.;
pub type Vec2D = nalgebra::Vector2<f32>;
pub type Vec3D = nalgebra::Vector3<f32>;

use na::{Isometry3, Matrix4, Point3, Quaternion, RealField, Rotation3, Unit, Vector3, Vector4};
use nalgebra as na;

pub fn projection_transform(dw: f32, dh: f32) -> Matrix4<f32>
{
	Matrix4::new_perspective(dw / dh, f32::pi() / 2., 0.1, 2000.)
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

pub fn camera_project(x: f32, y: f32, z: f32, dir: f32) -> Matrix4<f32>
{
	let trans = Matrix4::new_translation(&Vector3::new(-x, -y, -z));
	let rot = Matrix4::from_axis_angle(&Unit::new_normalize(Vector3::new(1., 0., 0.)), -dir);

	rot * trans
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
