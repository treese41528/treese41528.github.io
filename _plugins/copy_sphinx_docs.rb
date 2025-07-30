# _plugins/copy_sphinx_docs.rb
require 'fileutils'

Jekyll::Hooks.register :site, :post_write do |site|
  # List of Sphinx‐generated directories (relative to your repo root) to copy over
  sphinx_dirs = [
    'VirtualHumans',
    'GenerativeAI_ComputationalPsychology',
    'STAT350/Website'
  ]

  sphinx_dirs.each do |rel_path|
    src  = File.join(site.source, rel_path)
    dest = File.join(site.dest,   rel_path)
    next unless Dir.exist?(src)

    # Remove any old copy in the generated site
    FileUtils.rm_rf(dest)
    # Recreate the folder in _site
    FileUtils.mkdir_p(dest)
    # Copy everything from the Sphinx output
    FileUtils.cp_r(Dir["#{src}/*"], dest)

    # Fix up any _static URLs in the copied HTML
    Dir.glob("#{dest}/**/*.html").each do |html_file|
      text = File.read(html_file)
      # Replace href="_static/... and src="_static/... with absolute paths
      text.gsub!(/(href|src)=['"]_static\//, "\\1=\"/#{rel_path}/_static/")
      File.write(html_file, text)
    end
  end
end
