# -----------------------------------------------------------------
# Central Limit Theorem Demonstration Shiny App (Smooth Curve for Population)
# Author: Timothy Reese
# Date: 02/04/2024
# Description: This Shiny app simulates the Central Limit Theorem.
# -----------------------------------------------------------------

library(shiny)
library(ggplot2)

# UI Definition
# -----------------------------------------------------------------
ui <- fluidPage(
  titlePanel("Central Limit Theorem Simulation"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("distribution", "Population Distribution",
                  choices = c("Normal", "Uniform", "Exponential", "Left-Skewed Beta", "Bimodal", "Binomial", "Poisson")),
      uiOutput("distInputs"), # Dynamic UI for distribution parameters
      numericInput("size", "Sample Size (n)", 1, step = 5),
      numericInput("numSamples", "Number of Samples", 1000, step = 500),
      actionButton("sample", "Simulate"),
      actionButton("resetButton", "Reset")
    ),
    
    mainPanel(
      plotOutput("populationPlot"),
      plotOutput("sampleMeansPlot"),
    )
  )
)

# Server Logic
# -----------------------------------------------------------------
server <- function(input, output, session) {
  
  # Dynamic UI for distribution parameters
  output$distInputs <- renderUI({
    switch(input$distribution,
           "Normal" = div(
             numericInput("mean", "Population Mean (μ)", 50),
             numericInput("sd", "Population SD (σ)", 10)
           ),
           "Uniform" = div(
             numericInput("min", "Minimum (a)", 0),
             numericInput("max", "Maximum (b)", 100)
           ),
           "Exponential" = numericInput("rate", "Rate (λ)", 1),
           "Left-Skewed Beta" = div(
             numericInput("alpha", "Alpha Parameter (α)", 15),
             numericInput("beta", "Beta Parameter (β)", 2)
           ),
           "Bimodal" = div(
             numericInput("mean1", "First Mean (μ1)", 20),
             numericInput("mean2", "Second Mean (μ2)", 80),
             numericInput("sd1", "First Standard Deviation (σ1)", 10),
             numericInput("sd2", "Second Standard Deviation (σ2)", 15),
             numericInput("proportion1", "Proportion of First Distribution", 0.5, min = 0, max = 1, step = 0.1)
           ),
           "Binomial" = div(
             numericInput("n", "Population Numer of Trials (n)", 15),
             numericInput("p", "Population Probability of Success (p)", 0.5)
           ),
           "Poisson" = numericInput("lambda", "Lambda (λ)", 3)
    )
  })
  
  # Improved function to determine the range dynamically
  calculate_poisson_range <- function(lambda) {
    if (lambda < 10) {
      return(0:qpois(0.999, lambda))  # Higher percentile for smaller lambdas
    }
    
    # For large lambdas, expand to ±3 standard deviations around the mean
    lower_bound <- max(0, round(lambda - 3 * sqrt(lambda)))
    upper_bound <- round(lambda + 3 * sqrt(lambda))
    
    return(seq(lower_bound, upper_bound))
  }
  
  # Improved function to determine the range dynamically
  calculate_binom_range <- function(n,p) {
    if (n < 15) {
      return(0:n)  # Higher percentile for smaller lambdas
    }
    
    # For large lambdas, expand to ±3 standard deviations around the mean
    lower_bound <- max(0, round(n*p - 3 * sqrt(n*(p*(1-p)))))
    upper_bound <- round(n*p + 3 *  sqrt(n*(p*(1-p))))
    
    return(seq(lower_bound, upper_bound))
  }
  
  
  # Helper function to generate data for the population plot
  population_density <- function(dist, params) {
    switch(dist,
           "Normal" = {
             mean <- params$mean
             sd <- params$sd
             x_vals <- seq(mean - 4 * sd, mean + 4 * sd, length.out = 1000)
             data.frame(x = x_vals, y = dnorm(x_vals, mean = mean, sd = sd))
           },
           "Uniform" = {
             min_val <- params$min
             max_val <- params$max
             x_vals <- seq(min_val, max_val, length.out = 1000)
             data.frame(x = x_vals, y = dunif(x_vals, min = min_val, max = max_val))
           },
           "Exponential" = {
             rate <- params$rate
             x_vals <- seq(0, qexp(0.99, rate), length.out = 1000)
             data.frame(x = x_vals, y = dexp(x_vals, rate = rate))
           },
           "Left-Skewed Beta" = {
             alpha <- params$alpha
             beta <- params$beta
             x_vals <- seq(0, 1, length.out = 1000)
             data.frame(x = x_vals, y = dbeta(x_vals, shape1 = alpha, shape2 = beta))
           },
           "Bimodal" = {
             mean1 <- params$mean1
             mean2 <- params$mean2
             sd1 <- params$sd1
             sd2 <- params$sd2
             proportion1 <- params$proportion1
             
             x_vals <- seq(min(mean1 - 4 * sd1, mean2 - 4 * sd2), max(mean1 + 4 * sd1, mean2 + 4 * sd2), length.out = 1000)
             y_vals <- proportion1 * dnorm(x_vals, mean = mean1, sd = sd1) +
               (1 - proportion1) * dnorm(x_vals, mean = mean2, sd = sd2)
             
             data.frame(x = x_vals, y = y_vals)
           },
           "Binomial" = {
             sample_size <- params$n
             probability <- params$p
             x_vals <- calculate_binom_range(sample_size, probability)
             data.frame(x = x_vals, y = dbinom(x_vals, size = sample_size, prob = probability))
           },
           "Poisson" = {
             lambda <- params$lambda
             x_vals <- calculate_poisson_range(lambda) 
             data.frame(x = x_vals, y = dpois(x_vals, lambda = lambda))
           },
           data.frame(x = numeric(0), y = numeric(0))  # Default empty data frame
    )
  }
  
  # Helper function to compute the population mean
  compute_population_mean <- function(dist, params) {
    switch(dist,
           "Normal" = params$mean,
           "Uniform" = (params$min + params$max) / 2,
           "Exponential" = 1 / params$rate,
           "Binomial" = params$n * params$p,
           "Poisson" = params$lambda,
           "Left-Skewed Beta" = params$alpha / (params$alpha + params$beta),  # Beta mean
           "Bimodal" = {
             # Weighted average of two distributions
             params$proportion1 * params$mean1 + (1 - params$proportion1) * params$mean2
           }
    )
  }
  
  populationMean <- reactiveVal(NA)
  
  
  # Helper function to compute the density value at the mean for the distribution
  density_at_mean <- function(dist, params) {
    switch(dist,
           "Normal" = dnorm(params$mean, mean = params$mean, sd = params$sd),
           "Uniform" = dunif((params$min + params$max) / 2, min = params$min, max = params$max),
           "Exponential" = dexp(1 / params$rate, rate = params$rate),
           "Binomial" = dbinom(round(params$n * params$p), size = params$n, prob = params$p),
           "Poisson" = dpois(params$lambda, lambda = params$lambda),
           "Left-Skewed Beta" = {
             mean_value <- params$alpha / (params$alpha + params$beta)
             dbeta(mean_value, shape1 = params$alpha, shape2 = params$beta)
           },
           "Bimodal" = {
             # Weighted density of two normal distributions at their means
             mean1_density <- dnorm(params$mean1, mean = params$mean1, sd = params$sd1)
             mean2_density <- dnorm(params$mean2, mean = params$mean2, sd = params$sd2)
             params$proportion1 * mean1_density + (1 - params$proportion1) * mean2_density
           },
           NA  # Return NA if unsupported
    )
  }
  
  # Helper function to draw a sample from a given distribution
  sample_population <- function(dist, n, params) {
    switch(dist,
           "Normal" = rnorm(n, mean = params$mean, sd = params$sd),
           "Uniform" = runif(n, min = params$min, max = params$max),
           "Exponential" = rexp(n, rate = params$rate),
           "Binomial" = rbinom(n, params$n, params$p),
           "Poisson" = rpois(n, lambda = params$lambda),
           "Left-Skewed Beta" = rbeta(n, shape1 = params$alpha, shape2 = params$beta),
           "Bimodal" = {
             mixture <- rbinom(n, 1, params$proportion1)
             rnorm(n, mean = ifelse(mixture == 1, params$mean1, params$mean2), 
                   sd = ifelse(mixture == 1, params$sd1, params$sd2))
           },
           numeric(0)  # Return empty numeric vector if unsupported
    )
  }
  
  # Reactive values to hold the population samples
  samples <- reactiveVal(list())
  is_discrete <- reactiveVal(FALSE)
  
  
  # Draw samples on button click
  observeEvent(input$sample, {
    dist <- input$distribution
    
    params <- list(mean = input$mean, sd = input$sd, min = input$min, max = input$max, rate = input$rate, n = input$n, p = input$p, lambda = input$lambda,
                   alpha = input$alpha, beta = input$beta, mean1 = input$mean1, mean2 = input$mean2, sd1 = input$sd1, sd2 = input$sd2, proportion1 = input$proportion1)
    
    n_replictions <- input$size
    numSamples <- input$numSamples
    
    mean_val <- compute_population_mean(dist, params)
    density_mean <- density_at_mean(dist, params)
    
    sampleMeans <- replicate(input$numSamples, {
      switch(dist,
             "Normal" = mean(sample_population(dist = "Normal", n = input$size, params = params)),
             "Uniform" = mean(sample_population(dist = "Uniform", n = input$size, params = params)),
             "Exponential" = mean(sample_population(dist = "Exponential", n = input$size, params = params)),
             "Binomial" = mean(sample_population(dist = "Binomial", n = input$size, params = params)),
             "Poisson" = mean(sample_population(dist = "Poisson", n = input$size, params = params)),
             "Left-Skewed Beta" = mean(sample_population(dist = "Left-Skewed Beta", n = input$size, params = params)),
             "Bimodal" = mean(sample_population(dist = "Bimodal", n = input$size, params = params)),
      )})
    
    is_discrete(dist %in% c("Poisson", "Binomial"))
    samples(list(population = population_density(dist, params), pop_means = mean_val, density_mean = density_mean, sampleMeans = sampleMeans))
  })
  
  
  
  # Plot the population distribution, handling discrete distributions separately
  output$populationPlot <- renderPlot({
    data <- samples()
    if (is.null(data$population) || nrow(data$population) == 0) return(NULL)
    
    if (is_discrete()) {
      # Convert x values to numeric for continuous scale
      data$population$x <- as.numeric(as.character(data$population$x))
      
      tick_interval <- ceiling(max(data$population$x) / 10)
      tick_labels <- seq(min(data$population$x), max(data$population$x), by = tick_interval)
      
      ggplot(data$population, aes(x = x, y = y)) +
        geom_bar(stat = "identity", fill = "purple", alpha = 1.0, linewidth = 2, color = "black") +
        geom_segment(aes(x = data$pop_means, xend = data$pop_means, y = 0, yend = data$density_mean), linetype = "dashed", color = "blue", lwd = 2) +
        labs(title = "Population Distribution (Discrete)", x = "", y = "Density") +
        scale_x_continuous(breaks = tick_labels) +
        theme_minimal() +
        theme(panel.background = element_rect(fill = 'white', color = 'black'),
              plot.title = element_text(size = 28, face = "bold"),
              axis.text.x = element_text(size = 24, face = "bold"),
              axis.text.y = element_text(size = 24, face = "bold"))
    } else {
      ggplot(data$population, aes(x = x, y = y)) +
        geom_area(fill = "purple", alpha = 1.0) +
        geom_line(color = "black", size = 2) +
        geom_segment(aes(x = data$pop_means, xend = data$pop_means, y = 0, yend = data$density_mean), linetype = "dashed", color = "blue", lwd = 2) +
        labs(title = "Population Distribution (Continuous)", x = "", y = "Density") +
        theme_minimal() +
        theme(panel.background = element_rect(fill = 'white', color = 'black'),
              plot.title = element_text(size = 28, face = "bold"),
              axis.text.x = element_text(size = 24, face = "bold"),
              axis.text.y = element_text(size = 24, face = "bold"))
    }
  })
  
  
  # Plot the sample means
  output$sampleMeansPlot <- renderPlot({
    data <- samples()
    if (is.null(data$sampleMeans)) return(NULL)
    ggplot(data.frame(SampleMeans = data$sampleMeans), aes(x = SampleMeans)) +
      geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "purple", color = "black", linewidth = 3) +
      labs(title = "Distribution of Sample Means", x = "", y = "Density") +
      theme_minimal()+
      theme(panel.background = element_rect(fill = 'white', color = 'black'),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            plot.title = element_text(size = 28, face = "bold"),
            axis.text.x =element_text(size=24, face = "bold"),
            axis.text.y =element_text(size=24, face = "bold"))
  })
  
  # Reset all reactive values
  observeEvent(input$resetButton, {
    samples(list())
  })
}

# Run the app
shinyApp(ui = ui, server = server)

