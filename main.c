
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int CHAR_BUFFER_SIZE = 1024;
char *DELIMITER = ",";
int VECTOR_WIDTH = 5;

double* lineToVector(char *line, int width)
{

    char *token = strtok(line, DELIMITER);

    double d[width];

    int count = 0;

    while (token && count < width)
    {

        d[count] = sscanf(token,"");

        printf("%s", token);

        token = strtok(NULL, DELIMITER);
        count++;
    }

    return d;
}

int fileLength(FILE *file)
{
    int length = 0;

    char lineBuffer[CHAR_BUFFER_SIZE];

    while (fgets(lineBuffer, CHAR_BUFFER_SIZE, file))
    {
        length++;
    }

    rewind(file);
    return length;
}

double *createEmptyDoubleVector(int x, int y)
{
    return malloc(x * y * sizeof(double));
}

void populateVectorFromFile(double *vector, FILE *file, char *delimiter)
{

    int line = 0;
}

int main(int argc, char *argv[])
{
    
    FILE *inputVectorFile = fopen("vectors.csv","r");

    int numVectors = fileLength(inputVectorFile);
    printf("Number of Vectors: %d\n", numVectors);

    double *vectors = createEmptyDoubleVector(numVectors, VECTOR_WIDTH);

    populateVectorFromFile(vectors, inputVectorFile, DELIMITER);

    for (int i = 0; i < numVectors; i++)
    {

        for (int j = 0; j < VECTOR_WIDTH; j++)
        {

            printf("{i=%d,j=%d} %f", i, j, *(vectors + i * VECTOR_WIDTH + j));
        }
    }

    printf("\n");

    char lineBuffer[CHAR_BUFFER_SIZE];

    while (fgets(lineBuffer, CHAR_BUFFER_SIZE, inputVectorFile))
    {
        printf("%s",lineBuffer);
        lineToVector(lineBuffer);
    }

    free(vectors);
    fclose(inputVectorFile);

    return EXIT_SUCCESS;
}