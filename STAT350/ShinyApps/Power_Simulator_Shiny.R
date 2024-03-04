# -----------------------------------------------------------------
# Power Analysis Visualization
# Author: Timothy Reese
# Date: 01/22/2024
# Description: This Shiny app demonstrates the relationship between
# Type I and Type II error, power, sample size, 
# and the null and alternative values.
# -----------------------------------------------------------------




library(shiny)
library(ggplot2)

# Define UI
ui <- fluidPage(
  titlePanel("Power Visualization Applet"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("nullMean", "Null Mean", min = 50, max = 80, value = 64),
      sliderInput("altMean", "Alternative Mean", min = 50, max = 80, value = 66),
      sliderInput("alpha", "Alpha", min = 0, max = 0.2, value = 0.05),
      sliderInput("sampleSize", "Sample Size", min = 2, max = 400, value = 35),
      sliderInput("stdDev", "Population Standard Deviation", min = 1, max = 10, value = 3),
      radioButtons("hypothesis", "Hypothesis Direction",
                   choices = list("Left-tailed" = "less", "Right-tailed" = "greater", "Two-tailed" = "two.sided"), selected = "greater")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Plot", plotOutput("powerPlot")),
        tabPanel("Summary", verbatimTextOutput("summary"))
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  calcReactive <- reactive({
    null_mean <- input$nullMean
    alt_mean <- input$altMean
    alpha <- input$alpha
    sample_size <- input$sampleSize
    std_dev <- input$stdDev
    std_error <- std_dev / sqrt(sample_size)
    delta <- (alt_mean - null_mean) / std_error
    
    # Calculate the critical Z value based on the hypothesis direction
    z_critical <- if (input$hypothesis == "less") {
      qnorm(alpha)
    } else if (input$hypothesis == "greater") {
      qnorm(1 - alpha)
    } else {
      qnorm(1 - alpha / 2)
    }
    
    # Adjust the cutoff based on the hypothesis direction
    cutoff <- null_mean + z_critical * std_error
    
    # Calculate power using the normal distribution
    power <- if (input$hypothesis == "less") {
      pnorm(cutoff, alt_mean, std_error)
    } else if (input$hypothesis == "greater") {
      1 - pnorm(cutoff, alt_mean, std_error)
    } else {
      (1 - pnorm(cutoff, alt_mean, std_error)) + pnorm(null_mean - z_critical * std_error, alt_mean, std_error)
    }
    
    beta <- 1 - power
    list(z_critical = z_critical, delta = delta, power = power, beta = beta, std_error = std_error, cutoff = cutoff)
  })
  
  output$powerPlot <- renderPlot({
    calcValues <- calcReactive()
    
    # Define limits for the plot
    x_min <- min(input$nullMean, input$altMean) - 4 * calcValues$std_error
    x_max <- max(input$nullMean, input$altMean) + 4 * calcValues$std_error
    x_values <- seq(x_min, x_max, length.out = 1000)
    
    # Define density functions for null and alternative distributions
    null_density <- dnorm(x_values, mean = input$nullMean, sd = calcValues$std_error)
    alt_density <- dnorm(x_values, mean = input$altMean, sd = calcValues$std_error)
    
    # Start the ggplot object
    p <- ggplot() +
      geom_line(aes(x = x_values, y = null_density), color = "black", lwd = 2) +
      geom_line(aes(x = x_values, y = alt_density), color = "black", lwd = 2) +
      labs(title = "Power Analysis", x = "", y = "Density") +
      theme_minimal()
    
  
    
    # Shading logic for Type I Error, Type II Error, and Power
    if (input$hypothesis == "less") {
      # Power region on alternative distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values <= calcValues$cutoff, alt_density, 0)), 
                           fill = "green", alpha = 0.5)
      # Type I Error region on null distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values <= calcValues$cutoff, null_density, 0)), 
                           fill = "red", alpha = 0.8)
      # Type II Error region on alternative distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values >= calcValues$cutoff, alt_density, 0)), 
                           fill = "purple", alpha = 0.8)
    } else if (input$hypothesis == "greater") {
      # Power region on alternative distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values >= calcValues$cutoff, alt_density, 0)), 
                           fill = "green", alpha = 0.5)
      # Type I Error region on null distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values >= calcValues$cutoff, null_density, 0)), 
                           fill = "red", alpha = 0.8)
      # Type II Error region on alternative distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values <= calcValues$cutoff, alt_density, 0)), 
                           fill = "purple", alpha = 0.8)
    } else { # two.sided
      lower_bound <- input$nullMean - calcValues$z_critical * calcValues$std_error
      upper_bound <- input$nullMean + calcValues$z_critical * calcValues$std_error
      # Power regions on alternative distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values <= lower_bound | x_values >= upper_bound, alt_density, 0)), 
                           fill = "green", alpha = 0.5)
      # Type I Error regions on null distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values <= lower_bound | x_values >= upper_bound, null_density, 0)), 
                           fill = "red", alpha = 0.8)
      # Type II Error region on alternative distribution
      p <- p + geom_ribbon(aes(x = x_values, ymin = 0, ymax = ifelse(x_values > lower_bound & x_values < upper_bound, alt_density, 0)), 
                           fill = "purple", alpha = 0.8)
    }
    
    # Add the rejection region lines
    p <- p + geom_vline(xintercept = calcValues$cutoff, linetype = "dashed", color = "black")
    if (input$hypothesis == "two.sided") {
      p <- p + geom_vline(xintercept = input$nullMean - calcValues$z_critical * calcValues$std_error, linetype = "dashed", color = "black")
    }
    
    
    # Create LaTeX-style labels for mu_0 and mu_a
    mu_0_label <- paste("mu[0] =", input$nullMean)
    mu_a_label <- paste("mu[a] =", input$altMean)
    
    # Add shorter vertical lines and LaTeX-style labels for mu_0 and mu_a on the plot
    line_height <- 0.05  # Adjust line height as needed
    
    p <- p + geom_segment(aes(x = input$nullMean, xend = input$nullMean, 
                              y = 0, yend = line_height), 
                          linetype = "longdash", color = "black", lwd = 2) +
      annotate("text", x = input$nullMean, y = line_height, label = "mu[0]", 
               vjust = -1, color = "black", parse = TRUE, size = 14)
    
    p <- p + geom_segment(aes(x = input$altMean, xend = input$altMean, 
                              y = 0, yend = line_height), 
                          linetype = "longdash", color = "black", lwd = 2) +
      annotate("text", x = input$altMean, y = line_height, label = "mu[a]", 
               vjust = -1, color = "black", parse = TRUE, size = 14)
    
    
    
    # Return the plot
    p + theme(panel.background = element_rect(fill = 'white', color = 'black'),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.border = element_blank(),
              plot.title = element_text(size = 28, face = "bold"),
              axis.text.x =element_text(size=24, face = "bold"))
  })
  
  
  
  output$summary <- renderText({
    calcValues <- calcReactive()
    paste("Type I Error (Alpha):", input$alpha,
          "\nType II Error (Beta):", calcValues$beta,
          "\nPower of the test:", calcValues$power)
  })
}



# Run the application
shinyApp(ui = ui, server = server)