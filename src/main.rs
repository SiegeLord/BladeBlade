#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![feature(backtrace)]

mod game_state;
mod error;
mod sfx;
mod sprite;
mod utils;

use crate::error::Result;
use crate::utils::{load_config, world_to_screen, Vec2D, DT};
use allegro::*;
use allegro_dialog::*;
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

	let mut state = game_state::GameState::new()?;
	//~ if options.play_music
	//~ {
		//~ state.sfx.play_music()?;
	//~ }

	if options.fullscreen
	{
		state.core.set_new_display_flags(FULLSCREEN_WINDOW);
	}

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

	timer.start();
	while !quit
	{
		if draw && queue.is_empty()
		{
			state.core.set_target_bitmap(Some(display.get_backbuffer()));
			state.core.clear_to_color(Color::from_rgb_f(0., 0., 0.2));

			state.core.flip_display();
		}

		let event = queue.wait_for_event();
		match event
		{
			Event::DisplayClose { .. } => quit = true,
			Event::TimerTick { .. } =>
			{
				//~ let start = state.core.get_time();
				//~ let end = state.core.get_time();
				//~ println!("{}", 1. / (end - start));
				state.tick += 1;
				state.sfx.update_sounds()?;
				draw = true;
			}
			_ => ()
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
