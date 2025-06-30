# -----------------------------------------------------------------
# Confidence Interval Simulation Shiny App
# Author: Timothy Reese
# Date: 01/13/2024
# Description: This Shiny app simulates confidence intervals for 
# different statistical distributions and confidence interval methods.
# -----------------------------------------------------------------



library(shiny)
library(ggplot2)
library(gridExtra) # for arranging multiple plots



# UI Definition
# -----------------------------------------------------------------
ui <- fluidPage(
  titlePanel("Simulating Confidence Intervals"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("method", "Method",
                  choices = c("z sigma known", "z sigma unknown (using s)", "t")),
      selectInput("distribution", "Distribution",
                  choices = c("Normal", "Uniform", "Exponential", "Poisson")),
      uiOutput("distInputs"), # This output will change dynamically
      numericInput("size", "Sample size (n)", 5, step = 5),
      numericInput("intervals", "Number of intervals", 50, step = 10),
      numericInput("confLevel", "Confidence Level", 95, min = 80, max = 99.99, step = 0.5),
      actionButton("sample", "Sample"),
      actionButton("resetButton", "Reset")
    ),
    
    mainPanel(
      plotOutput("confIntervalsPlot"),
      verbatimTextOutput("resultsText"),
      plotOutput("combinedPlots")
    )
  )
)



# Server Logic
# -----------------------------------------------------------------
server <- function(input, output, session) {
  
  # Define distribution parameter mapping and updates for UI
  output$distInputs <- renderUI({
    switch(input$distribution,
           "Normal" = div(
             numericInput("mean", "Population mean (μ)", 40),
             numericInput("sd", "Population SD (σ)", 30)
           ),
           "Uniform" = div(
             numericInput("min", "Minimum (a)", 0),
             numericInput("max", "Maximum (b)", 1)
           ),
           "Exponential" = numericInput("rate", "Rate (λ)", 1),
           "Poisson" = numericInput("lambda", "Lambda (λ)", 1)
    )
  })
  
  
  # Define reset function for reseting graphs and text
  resetAll <- function() {
    all_sample_means(data.frame(Means = numeric(0)))
    
    
    intervals_containing_mu$z_known <- 0
    intervals_containing_mu$z_unknown <- 0
    intervals_containing_mu$t <- 0
    
    running_intervals_containing_mu$z_known <- 0
    running_intervals_containing_mu$z_unknown <- 0
    running_intervals_containing_mu$t <- 0
    
    running_total$total_intervals <- 0
    
    output$confIntervalsPlot <- renderPlot({ NULL })
    output$combinedPlots <- renderPlot({ NULL })
    output$resultsText <- renderText({ "Press 'Sample' to generate confidence intervals." })
  }
  
  
  
  # Confidence Interval Calculation Function
  # Calculates confidence intervals based on method and sigma
  conf_interval_func <- function(x, method, sigma) {
    n <- input$size
    xbar <- mean(x)
    s <- sd(x)
    alpha <- (100 - input$confLevel) / 100
    if (method == "z sigma known") {
      # Use population standard deviation
      error_margin <- qnorm(1-alpha/2) * sigma / sqrt(n)
      lower_bound <- xbar - error_margin
      upper_bound <- xbar + error_margin
    } else {
      # Use sample standard deviation
      error_margin <- if (method == "t") {
        qt(1-alpha/2, df = n - 1) * s / sqrt(n)
      } else {
        qnorm(1-alpha/2) * s / sqrt(n)
      }
      lower_bound <- xbar - error_margin
      upper_bound <- xbar + error_margin
    }
    
    return(c(lower_bound, upper_bound))
  }
  
  # Interval Containment Calculation Function
  # Calculates the number of intervals containing the mean
  calcIntervalsContainingMean <- function(confIntervals, true_mean) {
    intervals_df <- data.frame(
      Lower = confIntervals[1, ],
      Upper = confIntervals[2, ],
      ContainsMean = confIntervals[1, ] <= true_mean &  confIntervals[2, ] >= true_mean
    )
    sum(intervals_df$ContainsMean)
  }
  

  # Additional Reactive Values for Running Averages and Totals
  running_averages <- reactiveValues(z_known = 0, z_unknown = 0, t = 0)
  running_intervals_containing_mu <- reactiveValues(z_known = 0, z_unknown = 0, t = 0)
  intervals_containing_mu <- reactiveValues(z_known = 0, z_unknown = 0, t = 0)
  running_total <- reactiveValues(total_intervals = 0)
  all_sample_means <- reactiveVal(data.frame(Means = numeric(0)))
  # Compute the mean and standard deviation from all_sample_means reactive value
  mean_of_all_samples <- reactive({
    mean(all_sample_means()$Means)
  })
  sd_of_all_samples <- reactive({
    sd(all_sample_means()$Means)
  })
  
  
  
  
  # Reactive expression for intervals containing mu and running totals
  currentResults <- reactive({
    selected_method <- input$method
    intervals_containing_mu <- switch(selected_method,
                                      "z sigma known" = intervals_containing_mu$z_known,
                                      "z sigma unknown (using s)" = intervals_containing_mu$z_unknown,
                                      "t" = intervals_containing_mu$t
    )
    total_intervals <- running_total$total_intervals
    percent_containing_mu <- 100 * round(intervals_containing_mu / input$intervals, 4)
    list(
      method = selected_method,
      intervals_containing_mu = intervals_containing_mu,
      total_intervals = total_intervals,
      percent_containing_mu = percent_containing_mu
    )
  })
  

  
  # Observe changes in the sample or method input
  observeEvent(c(input$sample, input$method), {
    
    currentResults()
  })
  
  # Define what happens when sample button is pressed.
  observeEvent(input$sample, {
    # Depending on the distribution name, set the distribution
    distribution_func <- switch(input$distribution,
                                "Normal" = rnorm,
                                "Uniform" = runif,
                                "Exponential" = rexp,
                                "Poisson" = rpois,
                                stop("Unsupported distribution")
    )
    
    # Depending on the distribution, set parameters for sampling
    params <- switch(input$distribution,
                     "Normal" = list(mean = input$mean, sd = input$sd),
                     "Uniform" = list(min = input$min, max = input$max),
                     "Exponential" = list(rate = input$rate),
                     "Poisson" = list(lambda = input$lambda)
    )
    
    
    
    
    computeIntervals <- function(i, distribution_func, conf_interval_func, input) {
      if (input$distribution == "Normal") {
        sample <- distribution_func(input$size, mean = input$mean, sd = input$sd)
        sigma = input$sd
        true_mean <- input$mean
      } else if (input$distribution == "Uniform") {
        sample <- distribution_func(input$size, min = input$min, max = input$max)
        sigma <- (input$max - input$min) / sqrt(12)
        true_mean <- (input$max + input$min)/2
      } else if (input$distribution == "Exponential") {
        sample <- distribution_func(input$size, rate = input$rate)
        sigma <- 1 / input$rate
        true_mean <- 1 / input$rate
      } else if (input$distribution == "Poisson") {
        sample <- distribution_func(input$size, lambda = input$lambda)
        sigma <- sqrt(input$lambda)
        true_mean <- input$lambda
      }
      sample_mean <- mean(sample)
      list(
        z_known = conf_interval_func(sample, "z sigma known", sigma),
        z_unknown = conf_interval_func(sample, "z sigma unknown (using s)", sigma),
        t = conf_interval_func(sample, "t", sigma),
        mu = true_mean,
        xbar = sample_mean
      )
    }
    
    
    
    
    # Use lapply to apply the function over the sequence of intervals
    results <- lapply(1:input$intervals, computeIntervals, distribution_func, conf_interval_func, input)
    
    # Extract the confidence intervals for each
    confIntervalsList_z_known <- sapply(results, function(x) x$z_known)
    confIntervalsList_z_unknown <- sapply(results, function(x) x$z_unknown)
    confIntervalsList_t <- sapply(results, function(x) x$t)
    true_mean <- sapply(results, function(x) x$mu)
    xbars <-sapply(results, function(x) x$xbar)
    
    
    # Compute averages and intervals containing mu for each method
    intervals_containing_mu_z_known <- calcIntervalsContainingMean(confIntervalsList_z_known, true_mean[1])
    intervals_containing_mu_z_unknown <- calcIntervalsContainingMean(confIntervalsList_z_unknown, true_mean[1])
    intervals_containing_mu_t <- calcIntervalsContainingMean(confIntervalsList_t, true_mean[1])
    
    intervals_containing_mu$z_known <- intervals_containing_mu_z_known
    intervals_containing_mu$z_unknown <- intervals_containing_mu_z_unknown
    intervals_containing_mu$t <- intervals_containing_mu_t
  
    
    # Update running totals for intervals containing mu
    
    
    running_intervals_containing_mu$z_known <- running_intervals_containing_mu$z_known + intervals_containing_mu_z_known
    running_intervals_containing_mu$z_unknown <- running_intervals_containing_mu$z_unknown + intervals_containing_mu_z_unknown
    running_intervals_containing_mu$t <- running_intervals_containing_mu$t + intervals_containing_mu_t
    running_total$total_intervals <- running_total$total_intervals + input$intervals
  
    
    all_sample_means(rbind(all_sample_means(), data.frame(Means = xbars)))

    
    
    
    
    # Function to render confidence interval plot based on selected method
    renderConfIntervalsPlot <- function(confIntervals) {
      intervals_df <- data.frame(
      Lower = confIntervals[1, ],
      Upper = confIntervals[2, ],
      ContainsMean = confIntervals[1, ] <= true_mean &  confIntervals[2, ] >= true_mean
    )
      ggplot(intervals_df, aes(ymin = Lower, ymax = Upper, x = 1:nrow(intervals_df), color = ContainsMean)) +
        geom_errorbar(width = 1) +
        scale_color_manual(values = c("TRUE" = "green", "FALSE" = "red")) +
        labs(title = "Confidence Intervals", x = "Interval Number", y = "Confidence Interval") +
        theme_minimal()
    }
    
    
    output$combinedPlots <- renderPlot({
      # Create the two histograms as ggplot objects
      
      xbar <- mean(xbars)
      s <- sd(xbars)
      recentSamplePlot <- ggplot(data.frame(Means = xbars), aes(x = Means)) +
        geom_histogram(aes(y = after_stat(density)), bins = 24, color = "black", fill = "purple") +
        stat_function(fun=dnorm, args=list(mean=xbar, sd=s), col="blue", lwd=1) +
        geom_density(col="red", lwd=1) + 
        geom_vline(aes(xintercept = true_mean[1]), color = "red", linetype = "dashed") +
        labs(title = "Most Recent Sample Means", x = "Sample Means", y = "Frequency") +
        theme_minimal()
      
      xbar <- mean_of_all_samples()
      s <- sd_of_all_samples()
      sampleMeansDistributionPlot <- ggplot(all_sample_means(), aes(x = Means)) +
        geom_histogram(aes(y = after_stat(density)), bins = 24, color = "black",fill = "green") +
        stat_function(fun=dnorm, args=list(mean=xbar, sd=s), col="blue", lwd=1) +
        geom_density(col="red", lwd=1) + 
        geom_vline(aes(xintercept = true_mean[1]), color = "red", linetype = "dashed") +
        labs(title = "Sampling Distribution of the Sample Mean", x = "Sample Means", y = "Frequency") +
        theme_minimal()
      
      # Combine the plots side by side using grid.arrange
      grid.arrange(recentSamplePlot, sampleMeansDistributionPlot, ncol = 2)
    })
                     
    
    
    
    # Render the plot based on selected method
    output$confIntervalsPlot <- renderPlot({
      switch(input$method,
             "z sigma known" = renderConfIntervalsPlot(confIntervalsList_z_known),
             "z sigma unknown (using s)" = renderConfIntervalsPlot(confIntervalsList_z_unknown),
             "t" = renderConfIntervalsPlot(confIntervalsList_t)
      )
    })
    
    
  })
 

  
  observeEvent(input$sample, {
  output$resultsText <- renderText({
    results <- currentResults()
    if (results$method == 'z sigma known'){
      current_running_containing_mu <-running_intervals_containing_mu$z_known
    }
    else if(results$method == 'z sigma unknown (using s)'){
      
      current_running_containing_mu <-running_intervals_containing_mu$z_unknown
    }
    else{
      
      current_running_containing_mu <-running_intervals_containing_mu$t
    }
    percent_total <- 100*round(current_running_containing_mu/results$total_intervals,4)
    paste(
      results$intervals_containing_mu, "out of", input$intervals,
      "intervals contain μ for method", results$method, " (",
      results$percent_containing_mu, "%)\n",
      "(Running Total)",  current_running_containing_mu,
      "out of", results$total_intervals, "intervals contain μ (", percent_total, "%)"
    )
  })
  })
  
  observeEvent(input$distribution, {
    resetAll()
  })
  # Reset button functionality
  observeEvent(input$resetButton, {
    resetAll()
  })
  
}

# Run the app
shinyApp(ui = ui, server = server)