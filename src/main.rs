#![feature(stdsimd)]

#[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
compile_error!("Need avx512f, avx512bw");

use core::arch::x86_64::*;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};

/// The divisor we use for fixed point conversion
const FIXED_POINT_SHIFT:   u32 = 5;
const FIXED_POINT_DIVISOR: i16 = 1 << FIXED_POINT_SHIFT;

/// A fixed point integer, converting to a float is done by dividing by
/// [`FIXED_POINT_DIVISOR`]
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug)]
struct Fxpt(i16);

/// Width of the internal game field
const GAME_FIELD_WIDTH:  Fxpt = Fxpt(400 * FIXED_POINT_DIVISOR);

/// Height of the internal game field
const GAME_FIELD_HEIGHT: Fxpt = Fxpt(300 * FIXED_POINT_DIVISOR);

/// Speed change upon input on each frame
const INPUT_IMPULSE: Fxpt = Fxpt(2 * FIXED_POINT_DIVISOR);

/// Player X coord
const PLAYER_X: Fxpt = Fxpt(112 * FIXED_POINT_DIVISOR);

/// Gravity the player experiences
const GRAVITY: Fxpt = Fxpt((1.6 * FIXED_POINT_DIVISOR as f32) as i16);

/// Friction the player experiences
const FRICTION: Fxpt = Fxpt((0.9 * FIXED_POINT_DIVISOR as f32) as i16);

/// Width and height dimension of the players collision square
const PLAYER_SIZE: Fxpt = Fxpt(4 * FIXED_POINT_DIVISOR);

/// The width of a wall or obstacle
const OBSTACLE_WIDTH: Fxpt = Fxpt(16 * FIXED_POINT_DIVISOR);

/// Number of pixels to scroll the screen by each frame
const SCROLL_SPEED: Fxpt = Fxpt(8 * FIXED_POINT_DIVISOR);

/// Number of walls which we generate ahead of time
/// This is the number of walls we can possibly collide with right away
const PREGEN_WALLS: usize = ((PLAYER_SIZE.0 + OBSTACLE_WIDTH.0 - 1) /
                              OBSTACLE_WIDTH.0) as usize;

/// Maximum number of frames
const MAX_FRAMES: usize = 10000;

/// Maximum number of objects our player can collide with.
/// This is the ceiling of the player size divided by the obstacle size which
/// gives us the number blocks we can directly collide with. We also can
/// collide with one block via straddling a center block so we add one extra.
/// We then multiply by two for the top and bottom walls, and add one
/// additional wall which is the obstacle in the center.
///
/// This logic is only incorrect if PLAYER_SIZE is 1, and thus we cannot
/// straddle a wall
const MAX_COLLISION_BLOCKS: usize = (PREGEN_WALLS + 1) * 2 + 1;

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug)]
struct Mask(__mmask32);

impl core::ops::Not for Mask {
    type Output = Self;

    fn not(self) -> Self::Output {
        unsafe { Self(_knot_mask32(self.0)) }
    }
}

impl core::ops::BitAnd for Mask {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl core::ops::BitOr for Mask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for Mask {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl From<i16> for Fxpt {
    fn from(val: i16) -> Self {
        Self(val * FIXED_POINT_DIVISOR)
    }
}

impl From<f32> for Fxpt {
    fn from(val: f32) -> Self {
        Self((val * FIXED_POINT_DIVISOR as f32) as i16)
    }
}

#[derive(Clone, Copy)]
struct VecFxpt(__m512i);

impl VecFxpt {
    /// Number of lanes per vector
    const LANES: usize = 32;

    fn random() -> Self {
        VecFxpt(unsafe {
            _mm512_set_epi64(
                rand::random(), rand::random(), rand::random(), rand::random(),
                rand::random(), rand::random(), rand::random(), rand::random(),
            )
        })
    }

    fn splat<V: Into<Fxpt>>(val: V) -> Self {
        unsafe { Self(_mm512_set1_epi16(val.into().0)) }
    }

    /// Compute the horizontal maximum, this is done very slowly so just use
    /// it sparingly
    fn hmax(self) -> (usize, i16) {
        // Save the vector to a temporary on the stack
        let mut tmp = [0i16; 32];
        unsafe {
            _mm512_storeu_epi16(tmp.as_mut_ptr(), self.0);
        }
        tmp.iter().copied().enumerate().max_by_key(|x| x.1).unwrap()
    }
    
    fn extract(self, idx: usize) -> i16 {
        // Save the vector to a temporary on the stack
        let mut tmp = [0i16; 32];
        unsafe {
            _mm512_storeu_epi16(tmp.as_mut_ptr(), self.0);
        }
        tmp[idx]
    }
    
    fn shr<const SHAMT: u32>(self) -> Self {
        unsafe { Self(_mm512_srai_epi16(self.0, SHAMT)) }
    }

    fn min<V: Into<VecFxpt>>(self, rhs: V) -> Self {
        unsafe { Self(_mm512_min_epi16(self.0, rhs.into().0)) }
    }

    fn max<V: Into<VecFxpt>>(self, rhs: V) -> Self {
        unsafe { Self(_mm512_max_epi16(self.0, rhs.into().0)) }
    }

    fn clamp<V: Into<VecFxpt>, Z: Into<VecFxpt>>(self, min: V, max: Z) -> Self{
        self.min(max).max(min)
    }
    
    fn cond_add<V: Into<VecFxpt>>(self, mask: Mask, rhs: V) -> Self {
        unsafe {
            Self(_mm512_mask_add_epi16(self.0, mask.0, self.0, rhs.into().0))
        }
    }

    fn cond_sub<V: Into<VecFxpt>>(self, mask: Mask, rhs: V) -> Self {
        unsafe {
            Self(_mm512_mask_sub_epi16(self.0, mask.0, self.0, rhs.into().0))
        }
    }

    fn cmp_le<V: Into<VecFxpt>>(self, rhs: V) -> Mask {
        unsafe { Mask(_mm512_cmple_epi16_mask(self.0, rhs.into().0)) }
    }
    
    fn cmp_lt<V: Into<VecFxpt>>(self, rhs: V) -> Mask {
        unsafe { Mask(_mm512_cmplt_epi16_mask(self.0, rhs.into().0)) }
    }
    
    fn cmp_gt<V: Into<VecFxpt>>(self, rhs: V) -> Mask {
        unsafe { Mask(_mm512_cmpgt_epi16_mask(self.0, rhs.into().0)) }
    }
    
    fn merge<V: Into<VecFxpt>>(self, mask: Mask, rhs: V) -> Self {
        unsafe { Self(_mm512_mask_mov_epi16(self.0, mask.0, rhs.into().0)) }
    }
}

impl<T: Into<Fxpt>> From<T> for VecFxpt {
    fn from(val: T) -> Self {
        Self::splat(val)
    }
}

impl core::fmt::Debug for VecFxpt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Save the vector to a temporary on the stack
        let mut tmp = [0i16; 32];
        unsafe {
            _mm512_storeu_epi16(tmp.as_mut_ptr(), self.0);
        }

        // Dump the vector as floats
        f.debug_list().entries(tmp.iter().map(|&x| {
            x as f32 / FIXED_POINT_DIVISOR as f32
        })).finish()
    }
}

impl<T: Into<VecFxpt>> core::ops::Add<T> for VecFxpt {
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        unsafe { Self(_mm512_add_epi16(self.0, other.into().0)) }
    }
}

impl<T: Into<VecFxpt>> core::ops::Sub<T> for VecFxpt {
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        unsafe { Self(_mm512_sub_epi16(self.0, other.into().0)) }
    }
}

impl<T: Into<VecFxpt>> core::ops::AddAssign<T> for VecFxpt {
    fn add_assign(&mut self, other: T) {
        *self = *self + other.into();
    }
}

impl<T: Into<VecFxpt>> core::ops::SubAssign<T> for VecFxpt {
    fn sub_assign(&mut self, other: T) {
        *self = *self - other.into();
    }
}

impl<T: Into<VecFxpt>> core::ops::Mul<T> for VecFxpt {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        unsafe { Self(_mm512_mullo_epi16(self.0, other.into().0)) }
    }
}

#[derive(Debug, Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new() -> Self {
        Self(rand::random())
    }

    fn seed(seed: u64) -> Self {
        Self(seed)
    }

    fn rand(&mut self) -> u64 {
        let ret = self.0;
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 43;
        ret
    }
}

#[derive(Clone, Copy, Debug)]
struct VecRng(__m512i);

impl VecRng {
    fn new() -> Self {
        Self(VecFxpt::random().0)
    }

    fn rand(&mut self) -> __m512i {
        unsafe {
            let ret = self.0;
            self.0 = _mm512_xor_epi64(self.0, _mm512_slli_epi64(self.0, 13));
            self.0 = _mm512_xor_epi64(self.0, _mm512_srli_epi64(self.0, 17));
            self.0 = _mm512_xor_epi64(self.0, _mm512_slli_epi64(self.0, 43));
            ret
        }
    }
}

#[derive(Debug)]
struct Game {
    /// Random number generator for the game
    rng: Rng,

    /// The player current movement speed
    player_speed: VecFxpt,
    player_y:     VecFxpt,

    // Bounds are [x1, x2) and [y1, y2)
    //
    // (x1, y1)
    // v
    // +-------+
    // |       |
    // |       |
    // +-------+
    //         ^ (x2, y2)

    blocks_x1: VecFxpt,
    blocks_y1: VecFxpt,
    blocks_x2: VecFxpt,
    blocks_y2: VecFxpt,

    wall_skew: Fxpt,
    
    dead: Mask,

    num_walls: u32,

    score: VecFxpt,

    inputs: Vec<VecFxpt>,

    vecrng:  VecRng,
    pathrng: Rng,
}

impl Game {
    fn new(map_seed: u64) -> Self {
        assert!(OBSTACLE_WIDTH.0 % SCROLL_SPEED.0 == 0,
            "Obstacle width not evenly divisble by scroll speed");

        assert!((GAME_FIELD_WIDTH.0 - PLAYER_X.0) % OBSTACLE_WIDTH.0 == 0,
            "Player X position must be an integer multiple from the right \
             wall of the obstacle width");

        let mut ret = Self {
            rng:          Rng::seed(map_seed),
            player_speed: VecFxpt::splat(0),
            player_y:     VecFxpt::splat(Fxpt(GAME_FIELD_HEIGHT.0 / 2)),
            blocks_x1:    VecFxpt::splat(-1),
            blocks_y1:    VecFxpt::splat(0),
            blocks_x2:    VecFxpt::splat(-1),
            blocks_y2:    VecFxpt::splat(0),
            wall_skew:    Fxpt(0),
            dead:         Mask(0),
            score:        VecFxpt::splat(0),
            num_walls:    0,
            inputs:       Vec::new(),
            vecrng:       VecRng::new(),
            pathrng:      Rng::new(),
        };

        for _ in 0..MAX_FRAMES {
            ret.inputs.push(VecFxpt::random());
        }

        for wall in 0..PREGEN_WALLS {
            ret.add_wall(wall as i16);
        }

        ret
    }

    fn restore(&mut self, best_score: usize, template: &Self) {
        self.rng = template.rng;
        self.player_speed = template.player_speed;
        self.player_y = template.player_y;
        self.blocks_x1 = template.blocks_x1;
        self.blocks_y1 = template.blocks_y1;
        self.blocks_x2 = template.blocks_x2;
        self.blocks_y2 = template.blocks_y2;
        self.wall_skew = template.wall_skew;
        self.dead = template.dead;
        self.score = template.score;
        self.num_walls = template.num_walls;
        self.inputs[best_score.saturating_sub(400)..best_score]
            .copy_from_slice(
                &template.inputs[best_score.saturating_sub(400)..best_score]);
    }

    fn add_wall(&mut self, offset: i16) {
        self.num_walls += 1;
        let gap_reduction = (self.num_walls / 20).min(180) as i16;
        let gap = Fxpt::from(250 - gap_reduction);
        
        let wall_size = Fxpt((GAME_FIELD_HEIGHT.0 - gap.0) / 2);

        self.wall_skew = Fxpt((self.wall_skew.0 +
            self.rng.rand() as i16 % (FIXED_POINT_DIVISOR * 8))
            .clamp(-wall_size.0, wall_size.0));

        // Find walls which are collideable
        let culled = self.blocks_x2.cmp_le(PLAYER_X);

        // Find the next two available slots for walls
        let top_wall = culled.0.trailing_zeros();
        let bot_wall = (culled.0 >> (top_wall + 1))
            .trailing_zeros() + top_wall + 1;
        let obs_wall = (culled.0 >> (bot_wall + 1))
            .trailing_zeros() + bot_wall + 1;
        
        let top_wall = Mask(1 << top_wall);
        let bot_wall = Mask(1 << bot_wall);
        let obs_wall = Mask(1 << obs_wall);

        let x1 = Fxpt(PLAYER_X.0 + OBSTACLE_WIDTH.0 * offset);
        let y1 = Fxpt(0);
        let x2 = Fxpt(x1.0 + OBSTACLE_WIDTH.0);
        let y2 = Fxpt(wall_size.0 + self.wall_skew.0);

        self.blocks_x1 = self.blocks_x1.merge(top_wall, x1);
        self.blocks_y1 = self.blocks_y1.merge(top_wall, y1);
        self.blocks_x2 = self.blocks_x2.merge(top_wall, x2);
        self.blocks_y2 = self.blocks_y2.merge(top_wall, y2); 
        
        let y1 = Fxpt(GAME_FIELD_HEIGHT.0 - (wall_size.0 -
                         self.wall_skew.0));
        let y2 = Fxpt(y1.0 + wall_size.0 - self.wall_skew.0);

        self.blocks_x1 = self.blocks_x1.merge(bot_wall, x1);
        self.blocks_y1 = self.blocks_y1.merge(bot_wall, y1);
        self.blocks_x2 = self.blocks_x2.merge(bot_wall, x2);
        self.blocks_y2 = self.blocks_y2.merge(bot_wall, y2);

        if self.num_walls % 4 == 0 { 
            let location = ((self.rng.rand() as u16) %
                (gap.0 - Fxpt::from(60).0) as u16) as i16;

            let y1 = Fxpt(wall_size.0 + self.wall_skew.0 + location);
            let y2 = Fxpt(y1.0 + Fxpt::from(60).0);

            self.blocks_x1 = self.blocks_x1.merge(obs_wall, x1);
            self.blocks_y1 = self.blocks_y1.merge(obs_wall, y1);
            self.blocks_x2 = self.blocks_x2.merge(obs_wall, x2);
            self.blocks_y2 = self.blocks_y2.merge(obs_wall, y2);
        }
    }

    fn run(&mut self, best_score: usize) {
        /// Number of frames of normal physics and collision before spawning
        /// a new wall
        const FRAMES_PER_WALL: i16 = OBSTACLE_WIDTH.0 / SCROLL_SPEED.0;

        for _ in 0..2 {
            let idx = best_score.saturating_sub(self.pathrng.rand() as usize % 400);
            let len = (self.pathrng.rand() as usize % 16)
                .min(self.inputs.len() - idx);

            match self.pathrng.rand() as u8 % 3 {
                0 => {
                    for idx in 0..len {
                        self.inputs[idx] = VecFxpt(self.vecrng.rand());
                    }
                }
                1 => {
                    self.inputs[idx..][..len].iter_mut()
                        .for_each(|x| *x = VecFxpt::splat(1));
                }
                2 => {
                    self.inputs[idx..][..len].iter_mut()
                        .for_each(|x| *x = VecFxpt::splat(-1));
                }
                _ => unreachable!(),
            }
        }

        let mut frames = 0usize;
        loop {
            self.add_wall(PREGEN_WALLS as i16);

            // Determine which walls will need to scroll
            let unculled = self.blocks_x2.cmp_gt(PLAYER_X);

            for _ in 0..FRAMES_PER_WALL {
                // Check for input
                let input_state = self.inputs[frames].cmp_lt(0);
                self.player_speed =
                    self.player_speed.cond_sub(input_state, INPUT_IMPULSE);

                // Move the map but only for blocks which are on the screen
                self.blocks_x1 =
                    self.blocks_x1.cond_sub(unculled, SCROLL_SPEED);
                self.blocks_x2 =
                    self.blocks_x2.cond_sub(unculled, SCROLL_SPEED);

                // Apply physics
                self.player_speed += GRAVITY;
                self.player_speed =
                    (self.player_speed.shr::<FIXED_POINT_SHIFT>()) * FRICTION;
            
                // Adjust player position
                self.player_y += self.player_speed;

                // Clamp the player to the screen
                self.player_y = self.player_y
                    .clamp(0, Fxpt(GAME_FIELD_HEIGHT.0 - PLAYER_SIZE.0));

                // Update the scores
                self.score = self.score.cond_add(!self.dead, Fxpt(1));

                // Check for collisions
                for ii in 0..MAX_COLLISION_BLOCKS {
                    unsafe {
                        let a1 = VecFxpt(_mm512_permutexvar_epi16(
                                VecFxpt::splat(Fxpt(ii as i16)).0,
                                self.blocks_x1.0));
                        let a2 = VecFxpt(_mm512_permutexvar_epi16(
                                VecFxpt::splat(Fxpt(ii as i16)).0,
                                self.blocks_x2.0));
                        let b1 = VecFxpt::splat(PLAYER_X);
                        let b2 = VecFxpt::splat(Fxpt(PLAYER_X.0 + PLAYER_SIZE.0));

                        let c1 = VecFxpt(_mm512_permutexvar_epi16(
                                VecFxpt::splat(Fxpt(ii as i16)).0,
                                self.blocks_y1.0));
                        let c2 = VecFxpt(_mm512_permutexvar_epi16(
                                VecFxpt::splat(Fxpt(ii as i16)).0,
                                self.blocks_y2.0));
                        let d1 = self.player_y;
                        let d2 = self.player_y + PLAYER_SIZE;

                        self.dead |= a1.max(b1).cmp_lt(a2.min(b2)) &
                            c1.max(d1).cmp_lt(c2.min(d2));
                    }
                }
        
                // All players died
                if self.dead.0 == !0 { return; }

                // Index into input stream
                frames += 1;
            }
        }
    }
}

#[derive(Default)]
struct Statistics {
    games:  u64,
    frames: u64,
    score:  u64,
    die:    bool,
    inputs: Vec<VecFxpt>,
}

fn worker(map_seed: u64, stats: Arc<Mutex<Statistics>>) {
    const BATCH_SIZE: u64 = 10000;

    let mut best_score    = 0u64;
    let mut template_game = Game::new(map_seed);
    let mut game          = Game::new(map_seed);

    loop {
        let mut total_frames = 0u64;

        for _ in 0..BATCH_SIZE {
            // Create a new game and run
            game.restore(best_score as usize, &template_game);
            game.run(best_score as usize);

            // Update stats
            let (idx, score) = game.score.hmax();
            let score = score as u64;
            total_frames += score * VecFxpt::LANES as u64;

            if score > best_score {
                best_score = score;

                template_game.inputs.clear();
                for inp in game.inputs.iter() {
                    template_game.inputs.push(
                        VecFxpt::splat(Fxpt(inp.extract(idx))));
                }
            }
        }

        let mut stats = stats.lock().unwrap();
        stats.games  += BATCH_SIZE;
        stats.frames += total_frames;

        if best_score > stats.score {
            let mut tmp = Vec::new();
            for val in template_game.inputs.iter() {
                let input_state = val.cmp_lt(0);
                if input_state.0 == 0 {
                    tmp.push(b'0');
                } else {
                    tmp.push(b'1');
                };
            }
            std::fs::write("best.bin", &tmp).unwrap();

            stats.inputs = template_game.inputs.clone();
            stats.score  = best_score;
        } else {
            template_game.inputs.copy_from_slice(&stats.inputs);
            best_score = stats.score;
        }

        if stats.die {
            return;
        }
    }
}

fn main() {
    loop {
        let mut threads = Vec::new();
        let seed = 0x3abec1e6f745691e; //rand::random::<u64>();

        let stats = Arc::new(Mutex::new(Statistics::default()));

        for _ in 0..192 {
            let stats = stats.clone();
            threads.push(std::thread::spawn(move || worker(seed, stats)));
        }

        let it = Instant::now();
        let mut last = 0;
        loop {
            std::thread::sleep(Duration::from_millis(200));
            let elapsed = it.elapsed().as_secs_f64();

            let mut stats = stats.lock().unwrap();
            let mf = stats.frames as f64 / 1e6;
            print!("{:10.2} Mframes/sec | {:5} best score {:016x}\n",
                mf / elapsed, stats.score, seed);

            let extra = (stats.score > 1500) as u32 as f64 * 10.;

            if elapsed >= stats.score as f64 / 1e-5 + extra {
                stats.die = true;
                break;
            }

            last = stats.score;
        }

        threads.drain(..).for_each(|x| x.join().unwrap());
    }
}

