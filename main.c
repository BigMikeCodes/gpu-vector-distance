
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int CHAR_BUFFER_SIZE = 1024;
char *DELIMITER = ",";
int VECTOR_WIDTH = 5;

void lineToVector(char *line)
{

    char *token = strtok(line, DELIMITER);

    while (token)
    {

        printf("%s", token);

        token = strtok(NULL, DELIMITER);
    }
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

int main(int argc, char *argv[])
{
    
    FILE *inputVectorFile = fopen("vectors.csv","r");

    int numVectors = fileLength(inputVectorFile);
    printf("Number of Vectors: %d", numVectors);

    char lineBuffer[CHAR_BUFFER_SIZE];

    while (fgets(lineBuffer, CHAR_BUFFER_SIZE, inputVectorFile))
    {
        printf("%s",lineBuffer);
        lineToVector(lineBuffer);
    }
    
    fclose(inputVectorFile);

    return EXIT_SUCCESS;
}