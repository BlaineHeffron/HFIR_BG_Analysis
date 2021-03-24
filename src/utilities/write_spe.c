#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>

#ifndef VMS
#include <termios.h>
#endif
#include "util.h"

#ifdef HAVE_GNU_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

FILE *infile = 0, *cffile = 0;    /* command file flag and file descriptor*/
int  cflog = 0;                   /* command logging flag */

/* ======================================================================= */
int read_cmd_file(char *outstring, int maxretlen)
{
  /* return -1 if command file is not open for reading */
  if (!infile) return -1;
  /* otherwise read a line from the file and return the # of chars */
  return cask("?", outstring, maxretlen);
} /* read_cmd_file */

/* ======================================================================= */
int log_to_cmd_file(char *string)
{
  /* return 1 if command file is not open for logging */
  if (!cflog || !cffile) return 1;
  /* otherwise write string to the file */
  fprintf(cffile, "%s\n", string);
  return 0;
} /* log_to_cmd_file */

/* ======================================================================= */
int cask(char *mesag, char *ans, int mca)
{
  /* mesag:  question to be asked, should be less than 4096 chars
     ans:    answer recieved, maximum of mca chars will be modified
     mca:    max. number of characters asked for in answer
     returns number of characters received in answer */

  FILE *file;

#ifdef XWIN
  /* modified for use with minig */
  extern int get_focus(int *);
  extern int set_focus(int);
  static int focus = 0;
#endif

#ifdef VMS
  extern char getchar_vms(int echoflag);
#else
  struct termios s, t;
#endif
  int    nca, i;
  char   *wfr, *prompt, nope = '\0';
#ifdef HAVE_GNU_READLINE
  char   *inputline;
#endif
  static char r[4096];

  if (!(prompt = getenv("RADWARE_PROMPT"))) prompt = &nope;

  if (mca == 0) {
    tell("%s", mesag); /* write message */
    return 0;
  }

#ifdef XWIN
  /* set input focus to original window */
  if (focus && !infile) set_focus(focus);
#endif

  if (infile) {
    /* write message */
    tell("%s%s", mesag, prompt);
    /* read response string from input command file */
    if (!fgets(r, 256, infile)) {
      /* have reached end of command file */
      infile = 0;
      strncpy(ans, "CF END", mca);
      nca = 6;
      return nca;
    }
    /* if reading from a file, and we read a line "?",
       then go to interactive mode to get a real answer from the user */
    if (!strcmp(r, "?") || !strcmp(r, "?\n") || !strcmp(r, "?\r\n")) {
      tell("\n");
      file = infile;
      infile = 0;
      nca = cask(mesag, ans, mca);
      infile = file;
    }
  } else if (mca > 4                           /* will accept a long string */
	     || ((wfr = getenv("RADWARE_AWAIT_RETURN")) &&
		 (*wfr == 'Y' || *wfr == 'y')) /* or wants to wait for \n */
#ifndef VMS
	     || (tcgetattr(0, &s))             /* or cannot get term attr */
#endif
	     ) {
    /* read string from stdin */
#ifdef HAVE_GNU_READLINE
    if (strlen(mesag) + strlen(prompt) > 4095) {
      strncpy(r, mesag, 4095);
    } else{
      snprintf(r, 4095, "%s%s", mesag, prompt); /* save mesag + prompt in r */
    }
    inputline = readline(r); /* use readline write r and to read response */
    strncpy(r, inputline, 256);
    if (strlen(inputline)) add_history(inputline);
    free(inputline);
#else
    /* write message */
    tell("%s%s", mesag, prompt);
    /* read response string from input command file */
    fgets(r, 256, stdin);
#endif
  } else {
    /* write message */
    tell("%s%s", mesag, prompt);
    /* read chars one-at-a-time from stdin */
#ifdef VMS
    i = 0;
    while ((r[i++] = getchar_vms(1)) != '\n' && i < mca);
#else
    tcgetattr(0, &s);
    tcgetattr(0, &t);
    t.c_lflag &= ~ICANON;
    t.c_cc[VMIN] = 1;
    t.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &t);
    i = 0;
    while ((r[i++] = (char) getchar()) != '\n' && i < mca);
    tcsetattr(0, TCSANOW, &s);
#endif
    if (r[i-1] != '\n') tell("\n");
    r[i] = '\0';
  }
  if ((nca = strlen(r)) > mca) {
    nca = mca;
    r[mca] = '\0';
  }

#ifdef XWIN
  /* save information about current input focus window */
  if (!focus && !infile) get_focus(&focus);
#endif

  /* remove trailing blanks, \r or \n */
  while (nca > 0 &&
	 (r[nca-1] == ' ' || r[nca-1] == '\n' || r[nca-1] == '\r')) {
    r[--nca] = '\0';
  }

  /* if reading from command file, echo response to stdout */
  if (infile) tell("%s\n", r);
  /* if log command file open, copy response */
  if (cflog) fprintf(cffile, "%s\n", r);

  /* copy response to ans */
  strncpy(ans, r, mca);
  return nca;
} /* cask */

/* ====================================================================== */
int caskyn(char *mesag)
{
  /* mesag:     question to be asked (character string)
     returns 1: answer = Y/y/1   0: answer = N/n/0/<return> */

  char ans[16];

  while (cask(mesag, ans, 1)) {
    if (ans[0] == 'N' || ans[0] == 'n' || ans[0] == '0') return 0;
    if (ans[0] == 'Y' || ans[0] == 'y' || ans[0] == '1') return 1;
  }
  return 0;
} /* caskyn */
int askfn(char *ans, int mca,
	  const char *default_fn, const char *default_ext,
	  const char *fmt, ...)
{
  /* fmt, ...:  question to be asked
     ans:       answer recieved, maximum of mca chars will be modified
     mca:       max. number of characters asked for in answer
     returns number of characters received in answer */
  /* this variant of ask asks for filenames;
     default_fn  = default filename (or empty string if no default)
     default_ext = default filename extension (or empty string if no default) */

  va_list ap;
  char    q[4096];
  int     nca = 0, ncq, j;

  va_start(ap, fmt);
  ncq = vsnprintf(q, 4095, fmt, ap);
  va_end(ap);

  if (strlen(default_ext)) {
    j = snprintf(q+ncq, 4095-ncq, "\n   (default .ext = %s)", default_ext);
    ncq += j;
  }
  if (strlen(default_fn)) {
    snprintf(q+ncq, 4095-ncq, "\n   (rtn for %s)", default_fn);
  }

  nca = cask(q, ans, mca);
  if (strlen(default_fn) && nca == 0) {
    strncpy(ans, default_fn, mca);
    nca = strlen(default_fn);
  }
  setext(ans, default_ext, mca);

  return nca;
} /* askfn */

/* ====================================================================== */
int askyn(const char *fmt, ...)
{
  /* fmt, ...:  question to be asked (format string and variable no. of args)
     returns 1: answer = Y/y/1   0: answer = N/n/0/<return> */

  va_list ap;
  char    q[4096];

  va_start(ap, fmt);
  vsnprintf(q, 4095, fmt, ap);
  va_end(ap);

  return caskyn(q);
} /* askyn */

/* ====================================================================== */
int ask(char *ans, int mca, const char *fmt, ...)
{
  /* fmt, ...:  question to be asked
     ans:       answer recieved, maximum of mca chars will be modified
     mca:       max. number of characters asked for in answer
     returns number of characters received in answer */

  va_list ap;
  char    q[4096];

  va_start(ap, fmt);
  vsnprintf(q, 4095, fmt, ap);
  va_end(ap);
  return cask(q, ans, mca);
}

/* ====================================================================== */
void tell(const char *fmt, ...)
{
  /* fmt, ...:  string to be output (format string and variable no. of args) */

  va_list ap;

  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
} /* tell */

/* ====================================================================== */
void warn1(const char *fmt, ...)
{
  /* fmt, ...:  string to be output (format string and variable no. of args) */
  /* same as tell() for standard command-line programs
     but redefined elsewhere as a popup for GUI versions */

  va_list ap;

  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
} /* warn */

/* ====================================================================== */
void warn(const char *fmt, ...)
{
  /* fmt, ...:  string to be output (format string and variable no. of args) */

  va_list ap;
  char    ans[16];
  int     nca = 1;

  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  nca = ask(ans, 1, "Press any key to continue...\n");
} /* warn */

/* ====================================================================== */
int ask_selection(int nsel, int default_sel, const char **choices,
		 const char *flags, const char *head, const char *tail)
{
  /* returns item number selected from among nsel choices (counting from 0)
     default_sel = default choice number (used for simple <rtn>, no choice)
     choices = array of strings from which to choose
     flags = optional array of chars to use as labels for the choices
     head, tail = strings at start and end of question
  */

  char q[4096], f[48], ans[8];
  int  nca = 0, ncq = 0, i;


  if (nsel < 2 || nsel > 36) return 0;

  if (*head) ncq = snprintf(q, 4095, "%s\n", head);

  if (strlen(flags) >= nsel) {
    strncpy(f, flags, nsel);
  } else {
    for (i = 0; i < 10; i++) f[i] = (char) (48 + i);
    for (i = 10; i < nsel; i++) f[i] = (char) (55 + i);
  }

  if (*head) ncq = snprintf(q+ncq, 4095, "%s\n", head);
  ncq += snprintf(q+ncq, 4095-ncq, "Select %c for %s,\n", f[0], choices[0]);
  for (i = 1; i < nsel-1; i++) {
   ncq += snprintf(q+ncq, 4095-ncq, "       %c for %s,\n", f[i], choices[i]);
  }
  ncq += snprintf(q+ncq, 4095-ncq, "    or %c for %s.\n",
		  f[nsel-1], choices[nsel-1]);
  if (default_sel >= 0 && default_sel <= nsel)
    ncq += snprintf(q+ncq, 4095-ncq,
		    "   (Default is %s)\n", choices[default_sel-1]);
  if (*tail)
    ncq += snprintf(q+ncq, 4095-ncq, "%s\n", tail);
  ncq += snprintf(q+ncq, 4095-ncq, "  ...Your choice = ?");

  while (1) {
    nca = ask(ans, 1, q);
    if (nca == 0) return default_sel;
    for (i = 0; i < nsel; i++) {
      if (*ans == f[i] ||
	  (f[i] >= 'A' && f[i] <= 'Z' && *ans == f[i] + 'a' - 'A') ||
	  (f[i] >= 'a' && f[i] <= 'z' && *ans == f[i] + 'A' - 'a'))
	return i;
    }
  }
} /* ask_selection */
int file_error(char *error_type, char *filename)
{
  /* write error message */
  /* cannot perform operation error_type on file filename */

  if (strlen(error_type) + strlen(filename) > 58) {
    warn1("ERROR - cannot %s file\n%s\n", error_type, filename);
  } else {
    warn1("ERROR - cannot %s file %s\n", error_type, filename);
  }
  return 0;
}
int setext(char *filnam, const char *cext, int filnam_len)
{
  /* set default extension of filename filnam to cext
     leading spaces are first removed from filnam
     if extension is present, it is left unchanged
     if no extension is present, cext is used
     returned value pointer to the dot of the .ext
     cext should include the dot plus a three-letter extension */

  int nc, iext;

  /* remove leading spaces from filnam */
  nc = strlen(filnam);
  if (nc > filnam_len) nc = filnam_len;
  while (nc > 0 && filnam[0] == ' ') {
    memmove(filnam, filnam+1, nc--);
    filnam[nc] = '\0';
  }
  /* remove trailing spaces from filnam */
  while (nc > 0 && filnam[nc-1] == ' ') {
    filnam[--nc] = '\0';
  }
  /* look for file extension in filnam
     if there is none, put it to cext */
  iext = 0;
  if (nc > 0) {
    for (iext = nc-1;
	 (iext > 0 &&
	  filnam[iext] != ']' &&
	  filnam[iext] != '/' &&
	  filnam[iext] != ':');
	 iext--) {
      if (filnam[iext] == '.') return iext;
    }
    iext = nc;
  }
  strncpy(&filnam[iext], cext, filnam_len - iext - 1);
  return iext;
}
int put_file_rec(FILE *fd, void *data, int numbytes)
{
  /* write one fortran-unformatted style binary record into data */
  /* returns 1 for error */

#ifdef VMS  /* vms */
  int   j1;
  short rh[2];
  char  *buf;

  buf = data;
  j1 = numbytes;
  if (numbytes <= 2042) {
    rh[0] = numbytes + 2; rh[1] = 3;
  } else {
    rh[0] = 2044; rh[1] = 1;
    while (j1 > 2042) {
      if (fwrite(rh, 2, 2, fd) != 2 ||
	  fwrite(buf, 2042, 1, fd) != 1) return 1;
       rh[1] = 0; j1 -= 2042; buf += 2042;
    }
    rh[0] = j1 + 2; rh[1] = 2;
  }
  if (fwrite(rh, 2, 2, fd) != 2 ||
      fwrite(buf, j1, 1, fd) != 1) return 1;
  /* if numbytes is odd, write an extra (padding) byte */
  if (2*(numbytes>>1) != numbytes) {
    j1 = 0;
    fwrite(&j1, 1, 1, fd);
  }
    
#else /* unix */

  if (fwrite(&numbytes, 4, 1, fd) != 1 ||
      fwrite(data, numbytes, 1, fd) != 1 ||
      fwrite(&numbytes, 4, 1, fd) != 1) return 1;
#endif
  return 0;
}

int inq_file(char *filename, int *reclen)
{
  /* inquire for file existence and record length in longwords
     returns 0 for file not exists, 1 for file exists */

  int  ext;
  char jfile[80];
  struct stat statbuf;

  *reclen = 0;
  if (stat(filename, &statbuf)) return 0;

  ext = 0;
  strncpy(jfile, filename, 80);
  ext = setext(jfile, "    ", 80);
  if (!strcmp(&jfile[ext], ".mat") ||
      !strcmp(&jfile[ext], ".MAT") ||
      !strcmp(&jfile[ext], ".esc") ||
      !strcmp(&jfile[ext], ".ESC")) {
    *reclen = 2048;
  } else if (!strcmp(&jfile[ext], ".spn") ||
	     !strcmp(&jfile[ext], ".SPN") ||
	     !strcmp(&jfile[ext], ".m4b") ||
	     !strcmp(&jfile[ext], ".M4B") ||
	     !strcmp(&jfile[ext], ".e4k") ||
	     !strcmp(&jfile[ext], ".E4K")) {
    *reclen = 4096;
  } else if (!strcmp(&jfile[ext], ".cub") ||
	     !strcmp(&jfile[ext], ".CUB")) {
    *reclen = 256;
  } else if (!strcmp(&jfile[ext], ".2dp") ||
	     !strcmp(&jfile[ext], ".2DP")) {
    if (statbuf.st_size <= 0) {
      *reclen = 0;
    } else {
      *reclen = (int) (0.5 + sqrt((float) (statbuf.st_size/4)));
    }
  }
  return 1;
}
FILE *open_new_file(char *filename, int force_open)
{
  /* safely open a new file
     filename: name of file to be opened
     force_open = 0 : allow return value NULL for no file opened
     force_open = 1 : require that a file be opened */

  int  j, nc, jext, fn_error = 0;
  char tfn[80], *owf;
  FILE *file = NULL;

  strncpy(tfn, filename, 80);
  jext = setext(tfn, "", 80);

  while (1) {

#ifdef VMS
    fn_error = 0;
#else
    if ((fn_error = inq_file(filename, &j))) {
      /* file exists */
      owf = getenv("RADWARE_OVERWRITE_FILE");
      if (owf && (*owf == 'Y' || *owf == 'y')) {
	tell("Overwriting file %s\n", filename);
      } else if (owf && (*owf == 'N' || *owf == 'n')) {
	tell("File %s already exists.\n", filename);
	fn_error = 1;
      } else {
	fn_error = !askyn("File %s already exists - overwrite? (Y/N)", filename);
      }
    }
#endif

    /* open file w+ */
    if (!fn_error) {
      if ((file = fopen(filename, "w+"))) return file;
      file_error("open or overwrite", filename);
    }

    while (1) {
      nc = askfn(filename, 72, "", &tfn[jext], "New file name = ?");
      if (nc == 0 && !force_open) return NULL;
      if ((nc > 0 && filename[nc-1] != '/') ||
	  askyn("Are you sure you want file %s? (Y/N)", filename)) break;
    }
  }
}

int wspec(char *filnam, float *spec, int idim)
{
  /* write spectra in gf3 format
     filnam = name of file to be created and written
     spec = spectrum of length idim */

  char buf[32];
  int  j, c1 = 1, rl = 0;
  char namesp[8];
  FILE *file;

  j = setext(filnam, ".spe", 80);
  if (!(file = open_new_file(filnam, 0))) return 1;
  strncpy(namesp, filnam, 8);
  if (j < 8) memset(&namesp[j], ' ', 8-j);

  /* WRITE(1) NAMESP,IDIM,1,1,1 */
  /* WRITE(1) SPEC */
#define W(a,b) { memcpy(buf + rl, a, b); rl += b; }
  W(namesp,8); W(&idim,4); W(&c1,4); W(&c1,4); W(&c1,4);
#undef W
  if (put_file_rec(file, buf, rl) ||
      put_file_rec(file, spec, 4*idim)) {
    file_error("write to", filnam);
    fclose(file);
    return 1;
  }
  fclose(file);
  return 0;
} /* wspec */
