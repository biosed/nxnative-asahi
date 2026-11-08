// Keep SDL on the initial thread and expose the portable host as dm_main.
#include "dm.h"

int dm_main(int argc, char **argv);

int main(int argc, char **argv) {
    return dm_main(argc, argv);
}
