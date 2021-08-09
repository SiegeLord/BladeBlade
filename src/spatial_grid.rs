use crate::error::{Error, Result};
use nalgebra::Point2;

pub struct SpatialGrid<T>
{
	cells: Vec<Vec<T>>,
	cell_size: f32,
	min: Point2<f32>,
	max: Point2<f32>,
}

impl<T: Clone> SpatialGrid<T>
{
	pub fn new(min: Point2<f32>, max: Point2<f32>, cell_size: f32) -> Self
	{
		let w = max.x - min.x;
		let h = max.y - min.y;

		let num_cells = ((w / cell_size).floor() as i32 + 1) * ((h / cell_size).floor() as i32 + 1);

		let mut cells = Vec::with_capacity(num_cells as usize);
		for _ in 0..num_cells
		{
			cells.push(vec![])
		}

		Self {
			cells: cells,
			cell_size: cell_size,
			min: min,
			max: max,
		}
	}

	fn num_cells_horiz(&self) -> i32
	{
		let w = self.max.x - self.min.x;
		(w / self.cell_size).floor() as i32 + 1
	}

	fn num_cells_vert(&self) -> i32
	{
		let h = self.max.y - self.min.y;
		(h / self.cell_size).floor() as i32 + 1
	}

	pub fn insert(&mut self, val: T, min: Point2<f32>, max: Point2<f32>) -> Result<()>
	{
		let real_min = min;
		let real_max = max;

		let min = min - self.min;
		let max = max - self.min;

		let x_min = (min.x / self.cell_size).floor() as i32;
		let y_min = (min.y / self.cell_size).floor() as i32;
		let x_max = (max.x / self.cell_size).floor() as i32;
		let y_max = (max.y / self.cell_size).floor() as i32;

		if x_min < 0
			|| y_min < 0 || x_max >= self.num_cells_horiz()
			|| y_max >= self.num_cells_vert()
		{
			return Err(Error::new(
				format!(
					"Out of bounds: {}, {}, {}, {}",
					real_min, real_max, self.min, self.max
				),
				None,
			));
		}

		for y in y_min..y_max + 1
		{
			for x in x_min..x_max + 1
			{
				let num_cells_horiz = self.num_cells_horiz();
				self.cells[(y * num_cells_horiz + x) as usize].push(val.clone());
			}
		}
		Ok(())
	}

	pub fn query(&self, min: Point2<f32>, max: Point2<f32>) -> Result<impl Iterator<Item = &T>>
	{
		let min = min - self.min;
		let max = max - self.min;

		let x_min = (min.x / self.cell_size).floor() as i32;
		let y_min = (min.y / self.cell_size).floor() as i32;
		let x_max = (max.x / self.cell_size).floor() as i32;
		let y_max = (max.y / self.cell_size).floor() as i32;

		if x_min < 0
			|| y_min < 0 || x_max >= self.num_cells_horiz()
			|| y_max > self.num_cells_vert()
		{
			return Err(Error::new(
				format!(
					"Out of bounds: {:?}, {:?}, {:?}, {:?}",
					min, max, self.min, self.max
				),
				None,
			));
		}

		Ok((x_min..x_max + 1)
			.into_iter()
			.zip((y_min..y_max + 1).into_iter())
			.flat_map(move |(x, y)| self.cells[(y * self.num_cells_horiz() + x) as usize].iter()))
	}
}

#[test]
fn spatial_grid_test()
{
	let mut grid = SpatialGrid::<i32>::new(Point2::new(-10., -10.), Point2::new(20., 20.), 3.);

	grid.insert(0, Point2::new(-5., -5.), Point2::new(1., 1.))
		.unwrap();
	grid.insert(1, Point2::new(3., 3.), Point2::new(15., 4.))
		.unwrap();

	let mut returned: Vec<_> = grid
		.query(Point2::new(-10., -10.), Point2::new(0., 0.))
		.unwrap()
		.collect();
	returned.sort();
	returned.dedup();
	assert_eq!(returned.len(), 1);

	let mut returned: Vec<_> = grid
		.query(Point2::new(-10., -10.), Point2::new(20., 20.))
		.unwrap()
		.collect();
	returned.sort();
	returned.dedup();
	assert_eq!(returned.len(), 2);
}
