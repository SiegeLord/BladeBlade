use rand::prelude::*;

pub fn get_speech(level: i32, seed: u64) -> &'static str
{
	let mut rng = StdRng::seed_from_u64(seed + level as u64);
	if level == 2
	{
		let jokes = ["Laughter is the cure to all ills. Except death."];
		jokes[rng.gen_range(0..jokes.len())]
	}
	else if level == 3
	{
		let jokes = ["I stole all these jokes, by the way."];
		jokes[rng.gen_range(0..jokes.len())]
	}
	else if level < 5
	{
		let jokes = [
			"Why is all the code I write imperative? Because it's definitely not functional.",
			"To see the source of all bugs in your code, turn off your monitor and stare into it.",
			"Where do bad lights go? To prism... it's a light sentence.",
			"Why not play my other game, About Space?",
		];
		jokes[rng.gen_range(0..jokes.len())]
	}
	else
	{
		let jokes = [
			"E'th 2 is one of the most fun games I've created. Why not play it later as well?",
			"You can nerf my skills, but you can never nerf my soul.",
			"If you're not part of the solution, you're part of the precipitate.",
			"Statistics may be dull, but it has its moments.",
			"What's better than enchiladas? N + 1 chiladas.",
			"I don't trust geometry teachers. They are always plotting something.",
			"I should get back to leveling in a real game...",
			"Today I thought of a color that doesn't exist... but then I realized it was just a pigment of my imagination.",
		];
		jokes[rng.gen_range(0..jokes.len())]
	}
}
