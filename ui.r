ui = navbarPage("AC 3.0: Neural Networks",
                theme = shinytheme("journal"),
                header = tags$head(
                  tags$link(rel = "stylesheet",
                            type = "text/css",
                            href = "style.css") 
                ),
                
# introduction
  tabPanel("Introduction",
    fluidRow(
      withMathJax(),
        column(width = 6, 
          wellPanel(
            class = "scrollable-well",
            div(
              class = "html-fragment",
              includeHTML("text/introduction.html")
                 ))),
           column(width = 6,
                  align = "center",
                  img(src = "neural-network.png", height = "300px"),
                  br(),br(), br(),
                  img(src = "single-neuron.png", height = "250px")
            ))),

# first activity
  tabPanel("Feeding-Forward",
    fluidRow(
      column(width = 6,
        wellPanel(
          class = "scrollable-well",
          div(
            class = "html-fragment",
            includeHTML("text/activity1.html")
          ))),
      column(width = 6,
             align = "center",
             img(src = "xor.png", height = "125px"),
             br(), br(),
   radioButtons("image_choice1", label = "select neural network",
                choices = c("template" = "xor-network.png",
                            "initial set-up" = "xor-initial-network.png",
                            "after feed-forward" = "xor-after-feedforward.png"),
                inline = TRUE),
   br(),
   imageOutput("activity1a", height = "300px")

      ))),
      
 # second activity
    tabPanel("Backpropagating",
      fluidRow(
        column(width = 6,
          wellPanel(
            class = "scrollable-well",
            div(
              class = "html-fragment",
              includeHTML("text/activity2.html")
            ))),
        column(width = 6,
          align = "center",
          radioButtons("image_choice2", label = "select neural network",
          choices = c("after feed-forward" = "xor-after-feedforward.png",
                      "after backpropagation" = "xor-after-backpropogation.png"),
                       inline = TRUE),
          br(), br(),
          imageOutput("activity2a", height = "500px")
          
        ))),

  # third activity
    tabPanel("Training & Testing",
      fluidPage(
        column(width = 6,
          wellPanel(
            class = "scrollable-well",
            div(
              class = "html-fragment",
              includeHTML("text/activity3.html")
            ))),
        column(width = 6,
          align = "center",
          splitLayout(
            sliderInput("eta1", label = HTML("log(&eta;)"),
                       min = -3, max = 0, value = -3,
                       step = 0.25, width = "150px"),
            sliderInput("epochs1", label = "log(epochs)",
                       min = 1, max = 4, value = 1, 
                       step = 0.5, width = "150px"),
            radioButtons("ts",label = "unique training samples",
                       choices = c(1,4), selected = 1,
                       inline = FALSE),
            radioButtons("avg", label = "average error?",
                       choices = c("yes","no"), selected = "no",
                       inline = FALSE)
          ),
          plotOutput("activity3a", height = "300px"),
          br(), br(),
            radioButtons("test1", label = "test sample",
                         choices = c("1/0/1", 
                                     "1/1/0",
                                     "0/0/0",
                                     "0/1/1"),
                         inline = TRUE),
            verbatimTextOutput("prediction", placeholder = TRUE),
            verbatimTextOutput("error", placeholder = TRUE)
               ))),

# fourth activity

  tabPanel("Wrapping Up",
    fluidPage(
      column(width = 6,
        wellPanel(
          class = "scrollable-well",
          div(
            class = "html-fragment",
            includeHTML("text/wrapup.html")
          ))),
      column(width = 6,
        align = "center",
        img(src = "error-vs-steps-1.png", height = "200px"),
        img(src = "error-vs-steps-2.png", height = "200px"),
        img(src = "sigmoidal.png", height = "200px")
    )))


) # closing for user interface
