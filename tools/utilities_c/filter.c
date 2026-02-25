#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define DATE_PATH "log_wo_warmup.txt"
#define WRITE_PATH "log_wo_warmup_16.txt"
#define STEP 4 

int main()
{
    FILE* tp = fopen(WRITE_PATH,"w");
    fclose(tp);
    FILE* fp = fopen(DATE_PATH,"r");
    char line[10000];
    int index=0;
    while (fgets(line,sizeof(line),fp)!=NULL) {
        index++;
        if (index%STEP==0) {
            FILE* tp = fopen(WRITE_PATH,"a");
            fputs(line,tp);  
            fclose(tp);
        }
    }
    fclose(fp);
}
