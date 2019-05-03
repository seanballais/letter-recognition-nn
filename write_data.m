function write_data(data, fid)
  raw_data = data(:);
  line = sprintf('%f,', raw_data);
  line = line(1:end - 1); % Strip away the final comma.
  line = sprintf('%s\n', line); % Add a new line since we need to write the
                                % next data in a new line.
  fdisp(fid, line);
endfunction
