#include <mpi.h>
#include <stdio.h>

// Compile your
// mpicc -o probe ./deep_learning_at_scale/chapter_5/mpi_probe.cpp
// Run locally with 2 processor
// mpirun -np 2 ./probe
// Run with 2 host simulated locally using slots
// mpirun -np 2 -H localhost:1,localhost:2,localhost:3 ./probe
// mpirun -np 3 -H localhost:1,localhost:2,localhost:3 ./probe
// Run it for more
// mpirun -np 10 -H localhost:1,localhost:2,localhost:3 ./probe

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    int word_rank, world_size;
    float value;
    MPI_Comm_rank(MPI_COMM_WORLD, &word_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // printf("Process %s of word_rank %d, world_size: %d\n", processor_name, word_rank, world_size);
    printf("Process %s of word_rank %d/%d: Before MPI_Bcast, value is %f\n", processor_name, word_rank, world_size, value);

    if (word_rank == 0)
    {
        // If it is ranked 0 then it a root process. Read the value from
        // user input and this value as message to broadcast to other processors
        printf("I am the root Process! Enter your number to broadcast?\n");
        scanf("%f", &value);
    }

    // Each processor invokes MPI_Bcast, data in value is broadcasted from root processor
    // Resulting in every other process receiving this data in their variable value
    MPI_Bcast(&value, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    printf("process %s of word_rank %d/%d: After MPI_Bcast, value is %f\n", processor_name, word_rank, world_size, value);

    MPI_Finalize();

    return 0;
}