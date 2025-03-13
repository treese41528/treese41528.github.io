Jekyll::Hooks.register :site, :post_write do |site|
    if Dir.exist?('VirtualHumans')
      FileUtils.cp_r('VirtualHumans', site.config['destination'])
    end
  end