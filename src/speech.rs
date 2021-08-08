use rand::prelude::*;

pub fn get_speech(level: i32, seed: u64) -> &'static str
{
	let mut rng = StdRng::seed_from_u64(seed + level as u64);
	if level == 2
	{
		"Welcome challenger! I shall be the one challening your sanity with bad jokes!"
	}
	else if level == 3
	{
		let jokes = ["I stole all these jokes, by the way."];
		jokes[rng.gen_range(0..jokes.len())]
	}
	else if level < 10
	{
		let jokes = [
			"Why is all the code I write imperative? Because it's definitely not functional.",
			"To see the source of all bugs in your code, turn off your monitor and stare into it.",
		];
		jokes[rng.gen_range(0..jokes.len())]
	}
	else
	{
		""
	}
}
