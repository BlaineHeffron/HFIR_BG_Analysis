      SUBROUTINE WSPEC(FILNAM,SPEC,IDIM)
        !f2py intent(in) :: FILNAM, SPEC, IDIM
C         subroutine to write spectra in gf3 format....
C            FILNAM = name of file to be created and written....
C            SPEC = REAL spectrum of dimension IDIM....

      CHARACTER*(*) FILNAM
      REAL          SPEC(IDIM)
      INTEGER       IDIM
      CHARACTER*8   NAMESP


C this sets the default extension on the file name....
      CALL SETEXT(FILNAM,'.spe',J)
      NAMESP = FILNAM(1:8)
      IF (J.le.8) NAMESP(J:8) = ' '

C this opens a new file to receive the data....
      CALL OPEN_NEW_UNF(1,FILNAM,0,0)
      WRITE(1) NAMESP,IDIM,1,1,1
      WRITE(1) SPEC
      CLOSE(1)

      RETURN
      END