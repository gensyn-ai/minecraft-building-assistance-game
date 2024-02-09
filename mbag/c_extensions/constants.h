
/*
    A bunch of constants. These are checked against the corresponding Python ones when
    the C module is loaded.
*/
#define NUM_ACTION_TYPES 10
#define NUM_BLOCKS 10
#define NUM_CHANNELS (NUM_ACTION_TYPES + 2 * (NUM_BLOCKS - 1))
#define NOOP_CHANNEL 0
#define PLACE_BLOCK_CHANNEL 1
#define BREAK_BLOCK_CHANNEL (1 + NUM_BLOCKS)
#define MOVE_POS_X_CHANNEL (2 + NUM_BLOCKS)
#define MOVE_NEG_X_CHANNEL (3 + NUM_BLOCKS)
#define MOVE_POS_Y_CHANNEL (4 + NUM_BLOCKS)
#define MOVE_NEG_Y_CHANNEL (5 + NUM_BLOCKS)
#define MOVE_POS_Z_CHANNEL (6 + NUM_BLOCKS)
#define MOVE_NEG_Z_CHANNEL (7 + NUM_BLOCKS)
#define GIVE_BLOCK_CHANNEL (8 + NUM_BLOCKS)
#define NOOP 0
#define PLACE_BLOCK 1
#define BREAK_BLOCK 2
#define MOVE_POS_X 3
#define MOVE_NEG_X 4
#define MOVE_POS_Y 5
#define MOVE_NEG_Y 6
#define MOVE_POS_Z 7
#define MOVE_NEG_Z 8
#define GIVE_BLOCK 9
#define CURRENT_BLOCKS 0
#define PLAYER_LOCATIONS 4
#define NO_ONE 0
#define CURRENT_PLAYER 1
#define AIR 0
#define BEDROCK 1
#define MAX_PLAYER_REACH 4.5
