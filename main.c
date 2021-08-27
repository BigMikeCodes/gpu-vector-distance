
#include <stdlib.h>
#include <stdio.h>


int CHAR_BUFFER_SIZE = 1024;

int main(int argc, char *argv[])
{
    
    FILE *inputVectorFile = fopen("vectors.csv","r");

    char lineBuffer[CHAR_BUFFER_SIZE];

    while (fgets(lineBuffer, CHAR_BUFFER_SIZE, inputVectorFile))
    {
        printf("%s",lineBuffer);
    }
    
    fclose(inputVectorFile);

    return EXIT_SUCCESS;
}