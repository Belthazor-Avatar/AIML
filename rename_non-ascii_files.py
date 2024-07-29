def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def rename_files(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if not is_ascii(filename):
                basename, ext = os.path.splitext(filename)
                valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
                new_basename = ''.join(c if c in valid_chars else '_' for c in basename)
                new_filename = new_basename + ext

                counter = 1
                while new_filename in filenames or len(new_basename.strip()) == 0:
                    new_filename = f'{new_basename}_{counter}{ext}'
                    counter += 1

                os.rename(os.path.join(foldername, filename), os.path.join(foldername, new_filename))
                print(f'Renamed file: {os.path.join(foldername, new_filename)}')


