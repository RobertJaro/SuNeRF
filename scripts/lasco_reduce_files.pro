;+
; Calibrate LASCO level-0.5 files to level 1 with REDUCE_LEVEL_1, one file at a
; time; a file that fails is reported and skipped, and a file whose level-1
; output already exists in OUT is not processed again.
;
; REDUCE_LEVEL_1 reads the source code from the second character of the file
; name and accepts 1, 2, 3, m and d. The C2 image counter has run past 23999999,
; so recent C2 files are named 24xxxxxx and are rejected. Their source digit is
; set to 3, the code REDUCE_LEVEL_1 already accepts for the 23xxxxxx rollover;
; it keeps the level-0.5 pointing path in GET_SUN_CENTER (digit <= 3) and gives
; the output the input file name.
;-
pro lasco_reduce_files, files, out
  ; REDUCE_LEVEL_1 output name: source digit 2 becomes 5, 1/3/4 become 4.
  out_names = file_basename(files)
  for i = 0L, n_elements(files) - 1 do begin
    name = out_names[i]
    strput, name, (strmid(name, 1, 1) eq '2') ? '5' : '4', 1
    out_names[i] = name
  endfor
  done = file_test(out + path_sep() + out_names)
  print, 'LASCO_REDUCE_FILES: level-1 output exists for ' + strtrim(long(total(done)), 2) + $
    ' of ' + strtrim(n_elements(files), 2) + ' files in ' + out + '; these are skipped.'
  for i = 0L, n_elements(files) - 1 do begin
    if done[i] then continue
    catch, error
    if error ne 0 then begin
      catch, /cancel
      print, 'FAILED: ' + files[i] + ' (' + !error_state.msg + ')'
      continue
    endif
    image = lasco_readfits(files[i], hdr)
    name = hdr.filename
    if strmid(name, 1, 1) eq '4' then strput, name, '3', 1
    hdr.filename = name
    reduce_level_1, image, HDRIN=hdr, SAVEDIR=out
    catch, /cancel
  endfor
end
