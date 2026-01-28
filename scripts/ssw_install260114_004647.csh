#!/bin/csh -f
############################################################
# created via WWW at Wed Jan 14 00:46:49 2026
############################################################
setenv SSW $HOME/ssw
setenv ssw_host sohoftp.nascom.nasa.gov
setenv ssw_sw_host sohoftp.nascom.nasa.gov
setenv ssw_tar_old_style ""
setenv ssw_sw_sets "ssw_ssw_site ssw_packages_binaries ssw_packages_mjastereo ssw_packages_nrl ssw_packages_s3drs ssw_packages_sbrowser ssw_packages_sunspice ssw_psp_gen ssw_psp_wispr ssw_soho_eit ssw_soho_gen ssw_soho_lasco ssw_so_gen ssw_so_solohi ssw_so_spice ssw_so_stix ssw_ssw_gen ssw_ssw_iris ssw_stereo_gen ssw_stereo_secchi ssw_vobs_gen ssw_vobs_ontology ssw_vobs_vso"
setenv ssw_db_sets ""
setenv ssw_passive_ftp 
setenv ssw_noepsv 
setenv ssw_skip_sizechk 1
setenv sdo $SSW/sdo
setenv proba2 $SSW/proba2
setenv goesn $SSW/goesn
setenv goesr $SSW/goesr
setenv so $SSW/so
setenv psp $SSW/psp
echo "Using cURL in place of heritage ftp..."
mkdir -p $SSW/offline/swmaint/tar
cd  $SSW/offline/swmaint/tar
echo;echo ssw_install.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_install.tar.Z
echo;echo ssw_ssw_site.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_ssw_site.tar.Z
echo;echo ssw_packages_binaries.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_packages_binaries.tar.Z
echo;echo ssw_packages_mjastereo.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_packages_mjastereo.tar.Z
echo;echo ssw_packages_nrl.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_packages_nrl.tar.Z
echo;echo ssw_packages_s3drs.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_packages_s3drs.tar.Z
echo;echo ssw_packages_sbrowser.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_packages_sbrowser.tar.Z
echo;echo ssw_packages_sunspice.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_packages_sunspice.tar.Z
echo;echo ssw_psp_gen.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_psp_gen.tar.Z
echo;echo ssw_psp_wispr.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_psp_wispr.tar.Z
echo;echo ssw_soho_eit.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_soho_eit.tar.Z
echo;echo ssw_soho_gen.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_soho_gen.tar.Z
echo;echo ssw_soho_lasco.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_soho_lasco.tar.Z
echo;echo ssw_so_gen.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_so_gen.tar.Z
echo;echo ssw_so_solohi.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_so_solohi.tar.Z
echo;echo ssw_so_spice.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_so_spice.tar.Z
echo;echo ssw_so_stix.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_so_stix.tar.Z
echo;echo ssw_ssw_gen.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_ssw_gen.tar.Z
echo;echo ssw_ssw_iris.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_ssw_iris.tar.Z
echo;echo ssw_stereo_gen.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_stereo_gen.tar.Z
echo;echo ssw_stereo_secchi.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_stereo_secchi.tar.Z
echo;echo ssw_vobs_gen.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_vobs_gen.tar.Z
echo;echo ssw_vobs_ontology.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_vobs_ontology.tar.Z
echo;echo ssw_vobs_vso.tar.Z;curl  -O https://soho.nascom.nasa.gov/solarsoft/offline/swmaint/tar/ssw_vobs_vso.tar.Z
############################################################

#
#                               S.L.Freeland
#				27-May-1996 - adapt from ys_install for SSW
#
#; Name:    ssw_install
#;
#; Purpose: auto-ftp, then install or upgrade from compressed tar files
#;
#; Calling Sequence:
#;   ssw_install [/noftp] [/site] [/nodelete] [/nodbase] [/noexe] [/help]
#;
#; Optional Parameters (override defaults):
#;	/full	  - force full upgrade
#;	/noftp	  - dont ftp new files (use existing $SSW/offline/swmaint/tar)
#;      /site     - force installation of site branch
#;      /nodbase  - dont unpack data base files
#;      /nodelete - dont remove tar files
#;      /noexe    - inhibits execution of customized script
#;      /size	  - print installation disk size requirements and exit
#;      /ysrem	  - remove branches from ys PRIOR to ftp transfer
#;      /help     - print out this header and exit with no action
#;      /nomail   - if set, inhibit mail (in case system dependent problem)
#;
#; Notes:
#;      site branch - available for local routines and local cusomization
#;                    default is to not clobber this if it exists
#;      dbase files - these include Yohkoh ephemeris, Yohkoh event logs
#;                    goes events/data.... etc  These are highly recommended
#;                    for full SW function unless disk space is extremely
#;                    limited at your site
#;
#;History:      
#;	27-May-1996 - Samuel Freeland, adapted for SSW
#;		      ys->SSW, permit SW installation package from $ssw_host
#;		      (default host = sohoftp.nascom.nasa.gov)
#;      14-Jul-2005 - S.L.Freeland - check $ssw_passive_ftp
#;      20-Jul-2005 - S.L.Freeland - gunzip instead of uncompress
#;       4-may-2009 - S.L.Freeland - inhibit extended passive if:
#;                      1) $ssw_noesv -or- 2) uname=Darwin
#;

set curdir=`pwd`
setenv ssw_curdir $curdir
if (!($?SSW)) then
   if ($curdir:t == "tar") set curdir=$curdir:h
   if ($curdir:t == "swmaint") then
      setenv SSW $curdir:h
   else
      if( (!($?ys)) && (-e /ys) ) then
        setenv ys /ys
      else
         echo  Please define environmental variable SSW and retry
         echo  or run this from your tar file location...
         exit           #### unstructured exit
      endif
   endif
endif

# --- setup defaults---
setenv ssw_noftp  1	        # default is to ftp files required
setenv ssw_dosite 0            # instll ys_site
setenv ssw_dodata 1            # install any existing non-ys (ydb,etc)
setenv ssw_remove 1            # remove tar files after installation
setenv ssw_doexe  1            # execute customized script
setenv ssw_dohelp 0	        # dont show help
setenv ssw_full   0	        # default is incremental if available
setenv ssw_size   0		# if set, print size diagnostics and exit
setenv ssw_ysrem  0		# if set, remove ys before ftp transfer
setenv ssw_nomail 0             # if set, inhibit mail-> user and SSW
# ----------------------------------------------------------------------
# update defaults if site config file exists
if (-e $SSW/site/setup/ssw_install.config) source $$SSW/site/setup/ssw_install.config
# ----------------------------------------------------------------------
# --- handle parameters ---
foreach argx ($argv)
   switch ($argx)
      case full:		# force full upgrade (def=incremental)
      case /full:
         setenv ssw_full 1
	 breaksw
      case site:                # force site branch expansion
      case /site:
         setenv ssw_dosite 1
         breaksw
      case dbase:                # force data base expansion (default)
      case /dbase:
         setenv ssw_dodata 1
         breaksw
      case nodbase:		 # inhibit data base expansion
      case /nodbase:
         setenv ssw_dodata 0
         breaksw
      case delete:              # force tar file removal
      case /delete:
      case remove:
      case /remove:
         setenv ssw_remove 1
         breaksw
      case /nodelete:           # inhibit tar file removal
      case nodelete:
      case noremove:
      case /noremove:
          setenv ssw_remove 0
          breaksw
      case noexe:               # inhibit exectution of script
      case /noexec:
      case /noexe:
         setenv ssw_doexe 0
         breaksw
      case help:
      case /help:
         setenv ssw_dohelp 1
         breaksw
      case noftp:
      case /noftp:
	 setenv ssw_noftp 1
         breaksw
      case ysrem:
      case /ysrem:
         setenv ssw_ysrem 1
         breaksw
      case size:
      case /size:
         setenv ssw_size 1
         breaksw
      case /nomail:                             # Inhibit mail
         setenv ssw_nomail 1
         breaksw
      default:                                  # dont know this one
          echo Unrecognized option: $argx
	  echo Exiting....
	  exit					### unstructured exit
          breaksw
   endsw
end
# ------------------------------------------------------------------

if ($ssw_dohelp == 1) then
   cat $SSW/offline/swmaint/tar/ssw_install | grep "#;"
   exit
endif

# ----- get newest installation kit ------------
unset noclobber				# protect agains local environ
set workdir=$SSW/offline/swmaint/script
set host=`hostname`
mkdir -p $workdir

# slf 13-Jul-1994 - make boot tape (/noftp) compatible with network install

set instools=ssw_install.tar

if (!($?ssw_sw_host)) then
   setenv ssw_host "150.144.30.91"
endif

if (!($?user)) then
   set user=`whoami`
endif

if ( (-e ssw_install.tar.Z) && ($ssw_noftp == 1)) then
   cp -p ssw_install.tar.Z $workdir			# probably boot tape
   cd $workdir   
else
   cd $workdir
   set ftpins = $workdir/getins.ftp
   set passive = `printenv ssw_passive_ftp`
   set noepsv = ""
   set epsvenv = `printenv ssw_noepsv`
   set darwin = `uname | grep arwin`
   if ($#darwin > 0) set noepsv="1"
   if ($#epsvenv > 0) set noepsv="1"
echo "===== passive="$passive" ============"
   echo Generating ftp transfer script
   ########## build ftp script ########
   echo "open "$ssw_host     				>  $ftpins
   echo user ftp "$user""@""$host"			>> $ftpins
   if ($passive == "1") echo passive on			>> $ftpins
   if ($noepsv == "1")  echo  epsv4 off                 >> $ftpins
   echo binary		     				>> $ftpins
   echo cd /solarsoft/offline/swmaint/tar   		>> $ftpins
   echo get $instools".Z"    				>> $ftpins
   echo bye		     				>> $ftpins
   ####################################
   echo Starting ftp transfer of installation package: $instools".Z"
   ftp -in < $ftpins 
endif

if (!(-e $instools".Z")) then
   if ($ssw_noftp == 1) then
      echo "Could not find installation package:"$instools.".Z"
      echo "Contact freeland@penumbra.nascom.nasa.gov for guidence..."
   else
      echo "Trouble transfering installation package, try again later..."
   endif
endif

gunzip -f $instools".Z"
tar -xf $instools
# transfer control to $workdir/ssw_install.control
echo Transferring control to: $workdir/ssw_install.control
source $workdir/ssw_install.control
exit
