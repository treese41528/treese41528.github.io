Jekyll::Hooks.register :site, :post_write do |site|
  if Dir.exist?('VirtualHumans')
    FileUtils.cp_r('VirtualHumans', site.config['destination'])
  end

  if Dir.exist?('GenerativeAI_ComputationalPsychology')
    FileUtils.cp_r('GenerativeAI_ComputationalPsychology', site.config['destination'])
  end

  if Dir.exist?('STAT350/Website')
    FileUtils.cp_r('STAT350/Website', site.config['destination'])
  end
end
