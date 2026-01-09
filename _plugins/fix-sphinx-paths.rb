Jekyll::Hooks.register :site, :post_write do |site|
  # List of Sphinx output folders to fix
  %w(VirtualHumans GenerativeAI_ComputationalPsychology STAT350/Website ComputationalDataScience/Website).each do |folder|
    dir = File.join(site.config['destination'], folder)
    if Dir.exist?(dir)
      # Fix CSS and JS paths in all HTML files within the folder
      Dir.glob(File.join(dir, '**', '*.html')).each do |file|
        content = File.read(file)
        
        # Replace relative paths with absolute paths for _static assets
        content.gsub!('href="_static/', "href=\"/#{folder}/_static/")
        content.gsub!('src="_static/', "src=\"/#{folder}/_static/")
        
        File.write(file, content)
      end
    end
  end
end
