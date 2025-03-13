Jekyll::Hooks.register :site, :post_write do |site|
    vh_dir = File.join(site.config['destination'], 'VirtualHumans')
    if Dir.exist?(vh_dir)
      # Fix CSS and JS paths in all HTML files
      Dir.glob(File.join(vh_dir, '**', '*.html')).each do |file|
        content = File.read(file)
        
        # Replace relative paths with absolute paths
        content.gsub!('href="_static/', 'href="/VirtualHumans/_static/')
        content.gsub!('src="_static/', 'src="/VirtualHumans/_static/')
        
        File.write(file, content)
      end
    end
  end