use crate::error::Result;
use crate::utils::{clamp, load_sample, Vec2D};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

use allegro::*;
use allegro_acodec::*;
use allegro_audio::*;

use rand::prelude::*;

pub struct Sfx
{
	audio: AudioAddon,
	acodec: AcodecAddon,
	sink: Sink,
	stream: Option<AudioStream>,
	sample_instances: Vec<SampleInstance>,
	exclusive_sounds: Vec<String>,
	exclusive_instance: Option<SampleInstance>,

	samples: HashMap<String, Sample>,
}

impl Sfx
{
	pub fn new(core: &Core) -> Result<Sfx>
	{
		let audio = AudioAddon::init(&core)?;
		let acodec = AcodecAddon::init(&audio)?;
		let sink = Sink::new(&audio).map_err(|_| "Couldn't create audio sink".to_string())?;

		Ok(Sfx {
			audio: audio,
			acodec: acodec,
			sink: sink,
			sample_instances: vec![],
			stream: None,
			exclusive_instance: None,
			exclusive_sounds: vec![],
			samples: HashMap::new(),
		})
	}

	pub fn cache_sample<'l>(&'l mut self, name: &str) -> Result<&'l Sample>
	{
		Ok(match self.samples.entry(name.to_string())
		{
			Entry::Occupied(o) => o.into_mut(),
			Entry::Vacant(v) => v.insert(load_sample(&self.audio, name)?),
		})
	}

	pub fn get_sample<'l>(&'l self, name: &str) -> Option<&'l Sample>
	{
		self.samples.get(name)
	}

	pub fn update_sounds(&mut self) -> Result<()>
	{
		self.sample_instances.retain(|s| s.get_playing().unwrap());
		if let Some(ref stream) = self.stream
		{
			if !stream.get_playing()
			{
				self.play_music()?
			}
		}

		if !self.exclusive_sounds.is_empty()
		{
			let mut play_next_sound = true;
			if let Some(exclusive_instance) = &self.exclusive_instance
			{
				play_next_sound = !exclusive_instance.get_playing().unwrap();
			}
			if play_next_sound
			{
				let name = self.exclusive_sounds.pop().unwrap();
				self.cache_sample(&name)?;
				let sample = self.samples.get(&name).unwrap();
				let instance = self
					.sink
					.play_sample(
						sample,
						1.,
						None,
						thread_rng().gen_range(0.9..1.1),
						Playmode::Once,
					)
					.map_err(|_| "Couldn't play sound".to_string())?;
				self.exclusive_instance = Some(instance);
			}
		}

		Ok(())
	}

	pub fn play_sound(&mut self, name: &str) -> Result<()>
	{
		self.cache_sample(name)?;
		let sample = self.samples.get(name).unwrap();
		let instance = self
			.sink
			.play_sample(
				sample,
				1.,
				None,
				thread_rng().gen_range(0.9..1.1),
				Playmode::Once,
			)
			.map_err(|_| "Couldn't play sound".to_string())?;
		self.sample_instances.push(instance);
		Ok(())
	}

	pub fn play_positional_sound(
		&mut self, name: &str, sound_pos: Vec2D, camera_pos: Vec2D,
	) -> Result<()>
	{
		self.cache_sample(name)?;
		let sample = self.samples.get(name).unwrap();

		let dist = (sound_pos - camera_pos).norm();
		let volume = clamp(50000. / (dist * dist), 0., 1.);
		let pan = clamp((sound_pos.x - camera_pos.x) / 1000., -1., 1.);

		let instance = self
			.sink
			.play_sample(
				sample,
				volume,
				Some(pan),
				thread_rng().gen_range(0.9..1.1),
				Playmode::Once,
			)
			.map_err(|_| "Couldn't play sound".to_string())?;
		self.sample_instances.push(instance);
		Ok(())
	}

	pub fn play_exclusive_sound(&mut self, name: &str) -> Result<()>
	{
		self.exclusive_sounds.insert(0, name.to_string());
		Ok(())
	}

	pub fn play_music(&mut self) -> Result<()>
	{
		let mut new_stream = AudioStream::load(&self.audio, "data/music.ogg")
			.map_err(|_| "Couldn't load positive_energy.xm".to_string())?;
		new_stream.attach(&mut self.sink).unwrap();
		new_stream.set_playmode(Playmode::Once).unwrap();
		self.stream = Some(new_stream);
		Ok(())
	}
}
