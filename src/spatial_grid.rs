use crate::utils;
use nalgebra::{Point2, Vector2};

#[derive(Debug, Copy, Clone)]
pub struct Rect
{
	pub start: Point2<f32>,
	pub end: Point2<f32>,
}

impl Rect
{
	pub fn contains(&self, point: Point2<f32>) -> bool
	{
		point.x >= self.start.x
			&& point.x < self.end.x
			&& point.y >= self.start.y
			&& point.y < self.end.y
	}

	pub fn intersects_with_rect(&self, rect: Rect) -> bool
	{
		self.end.x > rect.start.x
			&& self.end.y > rect.start.y
			&& self.start.x < rect.end.x
			&& self.start.y < rect.end.y
	}

	pub fn intersects_with_segment(&self, start: Point2<f32>, end: Point2<f32>) -> bool
	{
		let v1 = self.start;
		let v2 = Point2::new(self.end.x, self.start.y);
		let v3 = self.end;
		let v4 = Point2::new(self.start.x, self.end.y);

		//~ dbg!(v1, v2, v3, v4);

		//~ dbg!(utils::intersect_segment_segment(v1, v2, start, end));
		//~ dbg!(utils::intersect_segment_segment(v2, v3, start, end));
		//~ dbg!(utils::intersect_segment_segment(v3, v4, start, end));
		//~ dbg!(utils::intersect_segment_segment(v4, v1, start, end));
		//~ dbg!(utils::is_inside_poly(&[v1, v2, v3, v4], start));

		utils::intersect_segment_segment(v1, v2, start, end)
			|| utils::intersect_segment_segment(v2, v3, start, end)
			|| utils::intersect_segment_segment(v3, v4, start, end)
			|| utils::intersect_segment_segment(v4, v1, start, end)
			|| utils::is_inside_poly(&[v1, v2, v3, v4], start)
	}
}

#[derive(Debug, Clone)]
pub struct Entry<T>
{
	pub rect: Rect,
	pub inner: T,
}

pub fn entry<T>(start: Point2<f32>, end: Point2<f32>, inner: T) -> Entry<T>
{
	Entry {
		rect: Rect {
			start: start,
			end: end,
		},
		inner: inner,
	}
}

#[derive(Debug, Clone)]
pub struct SpatialGrid<T>
{
	entries: Vec<Entry<T>>,
	cells: Vec<Vec<usize>>,
	width: usize,
	height: usize,

	cell_width: f32,
	cell_height: f32,
}

impl<T> SpatialGrid<T>
{
	pub fn new(width: usize, height: usize, cell_width: f32, cell_height: f32) -> Self
	{
		Self {
			entries: vec![],
			cells: vec![vec![]; width * height],
			width: width,
			height: height,
			cell_width: cell_width,
			cell_height: cell_height,
		}
	}

	pub fn push(&mut self, entry: Entry<T>)
	{
		let idx = self.entries.len();

		let (start_i, start_j) = self.index_from_point(entry.rect.start);
		let (end_i, end_j) = self.index_from_point(entry.rect.end);

		for j in start_j..=end_j
		{
			for i in start_i..=end_i
			{
				self.cells[i + j * self.width].push(idx);
			}
		}

		self.entries.push(entry);
	}

	fn index_from_point(&self, point: Point2<f32>) -> (usize, usize)
	{
		let i = (point.x / self.cell_width) as i64;
		let j = (point.y / self.cell_height) as i64;

		let i = utils::max(0, utils::min(i, self.width as i64 - 1));
		let j = utils::max(0, utils::min(j, self.height as i64 - 1));
		(i as usize, j as usize)
	}

	pub fn all_pairs(
		&self, filter_fn: impl Fn(&Entry<T>, &Entry<T>) -> bool,
	) -> Vec<(&Entry<T>, &Entry<T>)>
	{
		let mut ids = vec![];
		for (id1, entry1) in self.entries.iter().enumerate()
		{
			let (start_i, start_j) = self.index_from_point(entry1.rect.start);
			let (end_i, end_j) = self.index_from_point(entry1.rect.end);

			for j in start_j..=end_j
			{
				for i in start_i..=end_i
				{
					for &id2 in &self.cells[i + j * self.width]
					{
						if id1 == id2
						{
							continue;
						}
						if !filter_fn(&entry1, &self.entries[id2])
						{
							continue;
						}
						if entry1.rect.intersects_with_rect(self.entries[id2].rect)
						{
							if id1 > id2
							{
								ids.push((id1, id2))
							}
							else
							{
								ids.push((id2, id1))
							}
						}
					}
				}
			}
		}

		ids.sort();
		ids.dedup();

		let mut res = vec![];
		for (id1, id2) in ids
		{
			res.push((&self.entries[id1], &self.entries[id2]));
		}
		res
	}

	pub fn query_rect(
		&self, start: Point2<f32>, end: Point2<f32>, filter_fn: impl Fn(&Entry<T>) -> bool,
	) -> Vec<&Entry<T>>
	{
		let (start_i, start_j) = self.index_from_point(start);
		let (end_i, end_j) = self.index_from_point(end);

		let mut ids = vec![];
		for j in start_j..=end_j
		{
			for i in start_i..=end_i
			{
				for &id in &self.cells[i + j * self.width]
				{
					let rect = Rect {
						start: start,
						end: end,
					};
					let entry = &self.entries[id];
					if filter_fn(entry) && rect.intersects_with_rect(entry.rect)
					{
						ids.push(id);
					}
				}
			}
		}

		ids.sort();
		ids.dedup();

		let mut res = vec![];
		for id in ids
		{
			res.push(&self.entries[id]);
		}
		res
	}

	pub fn query_segment(
		&self, start: Point2<f32>, end: Point2<f32>, filter_fn: impl Fn(&Entry<T>) -> bool,
	) -> Vec<&Entry<T>>
	{
		let (start_i, start_j) = self.index_from_point(start);
		let (end_i, end_j) = self.index_from_point(end);

		let (start_i, end_i) = if start_i > end_i
		{
			(end_i, start_i)
		}
		else
		{
			(start_i, end_i)
		};
		let (start_j, end_j) = if start_j > end_j
		{
			(end_j, start_j)
		}
		else
		{
			(start_j, end_j)
		};

		//~ dbg!(start_i, start_j, end_i, end_j);

		let mut ids = vec![];
		for j in start_j..=end_j
		{
			for i in start_i..=end_i
			{
				let cell = &self.cells[i + j * self.width];
				let cell_start =
					Point2::new(i as f32 * self.cell_width, j as f32 * self.cell_height);
				let cell_rect = Rect {
					start: cell_start,
					end: cell_start + Vector2::new(self.cell_width, self.cell_height),
				};
				//~ dbg!(
				//~ i,
				//~ j,
				//~ start,
				//~ end,
				//~ cell_start,
				//~ cell_start + Vector2::new(self.cell_width, self.cell_height),
				//~ cell_rect.intersects_with_segment(start, end)
				//~ );
				if cell_rect.intersects_with_segment(start, end)
				{
					for &id in cell
					{
						let entry = &self.entries[id];
						//~ dbg!(id, entry.rect, entry.rect.intersects_with_segment(start, end));
						if filter_fn(entry) && entry.rect.intersects_with_segment(start, end)
						{
							ids.push(id);
						}
					}
				}
			}
		}

		ids.sort();
		ids.dedup();

		let mut res = vec![];
		for id in ids
		{
			res.push(&self.entries[id]);
		}
		res
	}
}

#[test]
fn test_rect_segment()
{
	let rect = Rect {
		start: Point2::new(0., 64.),
		end: Point2::new(64., 128.),
	};
	let start = Point2::new(0., 256.);
	let end = Point2::new(0., 120.);

	assert!(rect.intersects_with_segment(start, end));
}
