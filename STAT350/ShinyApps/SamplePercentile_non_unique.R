# -----------------------------------------------------------------
# Sample Percentile Method Exploration
# Author: Timothy Reese
# Date: 01/17/2024
# Description: This Shiny app lets one explore the different methods
# that exist in the base R software for computing the sample quantiles.
# The observations are plotted as a dot plot with the sample quantiles
# added to the graph.
# -----------------------------------------------------------------


library(shiny)
library(ggplot2)
library(dplyr)

# Predefined values based on your specification
predefined_values <- c(5, 10, 25, 50, 55, 60, 80, 90, 95, 100)
predefined_freq <- c(10, 10, 6, 5, 2, 2, 5, 10, 10, 10)

# Define the UI
ui <- fluidPage(
  titlePanel("Percentile Calculation with Different Methods"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("percentile", "Percentile:", min = 0, max = 100, value = 50),
      lapply(1:length(predefined_values), function(i) {
        fluidRow(
          column(3, numericInput(paste0("val", i), label = paste("Value", i), value = predefined_values[i])),
          column(3, numericInput(paste0("freq", i), label = paste("Frequency", i), value = predefined_freq[i]))
        )
      }),
      textOutput("totalSamples")
    ),
    mainPanel(
      plotOutput("dotPlot")
    )
  )
)

# Define the server logic
server <- function(input, output) {
  # Function to calculate total samples
  calculateTotalSamples <- function() {
    total_samples <- sum(unlist(lapply(1:length(predefined_values), function(i) {
      input[[paste0("freq", i)]]
    })))
    return(total_samples)
  }
  
  # Reactive expression to automatically update data
  calculatedData <- reactive({
    nums <- unlist(lapply(1:length(predefined_values), function(i) {
      rep(input[[paste0("val", i)]], input[[paste0("freq", i)]])
    }))
    percentiles <- sapply(1:7, function(i) quantile(nums, probs = input$percentile/100, type = i))
    percentiles_df <- data.frame(Type = 1:7, Percentile = unname(percentiles))
    percentiles_df
  })
  
  
  # Update total samples when frequencies change
  observeEvent(lapply(1:length(predefined_values), function(i) input[[paste0("freq", i)]]), {
    total_samples <- calculateTotalSamples()
    output$totalSamples <- renderText({
      paste("Total Samples: ", total_samples)
    })
  })
  
  
  output$dotPlot <- renderPlot({
    data <- calculatedData()
    nums <- unlist(lapply(1:length(predefined_values), function(i) {
      rep(input[[paste0("val", i)]], input[[paste0("freq", i)]])
    }))
    
    # Create data frame for percentiles
    percentile_labels <- data %>%
      group_by(Percentile) %>%
      summarize(Types = paste(Type, collapse = ", "))
    
    ggplot() +
      geom_dotplot(data = data.frame(x = nums), aes(x = x), method = 'histodot', binwidth = 3, color = 'black', fill = 'purple') +
      geom_vline(data = percentile_labels, aes(xintercept = Percentile, color = as.factor(Types)), linetype = "dashed", lwd = 2) +
      scale_x_continuous(breaks = seq(0, 101, by = 5)) +
      labs(title = "Dot Plot with Percentiles",
           x = "Values",
           y = "Frequency",
           color = "Percentile Types") +
      theme_minimal()
  })
  
  # Display the total number of samples
  output$totalSamples <- renderText({
    total_samples <- calculateTotalSamples()
    paste("Total Samples: ", total_samples)
  })
}

# Run the application 
shinyApp(ui, server)
