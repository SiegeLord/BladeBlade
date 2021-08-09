#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![feature(backtrace)]

mod error;
mod game_state;
mod map;
mod menu;
mod sfx;
mod spatial_grid;
mod speech;
mod sprite;
mod utils;

use crate::error::Result;
use crate::game_state::{GameState, NextScreen};
use crate::utils::{load_config, DT};
use allegro::*;
use allegro_dialog::*;
use rand::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
struct Options
{
	fullscreen: bool,
	width: i32,
	height: i32,
	play_music: bool,
}

fn real_main() -> Result<()>
{
	let options: Options = load_config("options.cfg")?;

	let mut state = GameState::new()?;
	//~ if options.play_music
	//~ {
	//~ state.sfx.play_music()?;
	//~ }

	if options.fullscreen
	{
		state.core.set_new_display_flags(FULLSCREEN_WINDOW);
	}

	state.core.set_new_display_option(
		DisplayOption::DepthSize,
		16,
		DisplayOptionImportance::Suggest,
	);
	let display = Display::new(&state.core, options.width, options.height)
		.map_err(|_| "Couldn't create display".to_string())?;

	let timer =
		Timer::new(&state.core, DT as f64).map_err(|_| "Couldn't create timer".to_string())?;

	let queue =
		EventQueue::new(&state.core).map_err(|_| "Couldn't create event queue".to_string())?;
	queue.register_event_source(display.get_event_source());
	queue.register_event_source(
		state
			.core
			.get_keyboard_event_source()
			.expect("Couldn't get keyboard"),
	);
	queue.register_event_source(
		state
			.core
			.get_mouse_event_source()
			.expect("Couldn't get mouse"),
	);
	queue.register_event_source(timer.get_event_source());

	let mut quit = false;
	let mut draw = true;
	let mut rng = thread_rng();

	let mut menu = menu::Menu::new(
		&mut state,
		display.get_width() as f32,
		display.get_height() as f32,
	)?;
	let mut map: Option<map::Map> = None;
	let mut logics_without_draw = 0;

	timer.start();
	while !quit
	{
		if draw && queue.is_empty()
		{
			let start = state.core.get_time();
			state.core.set_target_bitmap(Some(display.get_backbuffer()));
			state.core.clear_to_color(Color::from_rgb_f(0., 0., 0.2));
			state.core.clear_depth_buffer(1.);

			if let Some(map) = &mut map
			{
				map.draw(&state)?;
			}
			else
			{
				menu.draw(&state)?;
			}

			state.core.flip_display();
			let end = state.core.get_time();
			if state.tick % 20 == 0
			{
				println!("FPS: {}", 1. / (end - start));
			}
			logics_without_draw = 0;
		}

		let event = queue.wait_for_event();
		let next_screen;
		if let Some(map) = &mut map
		{
			next_screen = map.input(&event, &mut state)?;
		}
		else
		{
			next_screen = menu.input(&event, &mut state)?;
		}
		if let Some(next_screen) = next_screen
		{
			match next_screen
			{
				NextScreen::Game =>
				{
					map = Some(map::Map::new(
						&mut state,
						rng.gen_range(0..16000),
						display.get_width() as f32,
						display.get_height() as f32,
					)?);
				}
				NextScreen::Menu =>
				{
					map = None;
				}
				NextScreen::Quit =>
				{
					quit = true;
				}
			}
		}
		match event
		{
			Event::DisplayClose { .. } => quit = true,
			Event::TimerTick { .. } =>
			{
				//~ let start = state.core.get_time();
				//~ let end = state.core.get_time();
				//~ println!("{}", 1. / (end - start));
				if let Some(map) = &mut map
				{
					if logics_without_draw < 10
					{
						let start = state.core.get_time();
						map.logic(&mut state)?;
						let end = state.core.get_time();
						if state.tick % 20 == 0
						{
							println!("LPS: {}", 1. / (end - start));
						}
						logics_without_draw += 1;
					}
				}
				else
				{
					menu.logic(&mut state)?;
				}
				state.sfx.update_sounds()?;

				if !state.paused
				{
					state.tick += 1;
				}
				draw = true;
			}
			_ => (),
		}
	}

	Ok(())
}

fn main()
{
	use std::panic::catch_unwind;

	match catch_unwind(|| real_main().unwrap())
	{
		Err(e) =>
		{
			let err: String = e
				.downcast_ref::<&'static str>()
				.map(|&e| e.to_owned())
				.or_else(|| e.downcast_ref::<String>().map(|e| e.clone()))
				.unwrap_or("Unknown error!".to_owned());

			//~ error!("{}", err);
			show_native_message_box(
				None,
				"Error!",
				"An error has occurred!",
				&err,
				Some("You make me sad."),
				MESSAGEBOX_ERROR,
			);
		}
		Ok(_) => (),
	}
}
