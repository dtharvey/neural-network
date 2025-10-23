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
  tabPanel("Feed-Forward",
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
   radioButtons("image_choice1", label = "select neural network",
                choices = c("template" = "template.png",
                            "initial set-up" = "initial.png",
                            "after feed-forward" = "final.png"),
                inline = TRUE),
   br(), br(),
   imageOutput("activity1a", height = "500px")

      ))),
      
 # second activity
    tabPanel("Backpropogation",
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
                       choices = c("after feed-forward" = "final.png",
                                   "after backpropogation" = "backprop.png"),
                       inline = TRUE),
          br(), br(),
          imageOutput("activity2a", height = "500px")
          
        ))),

  # third activity
    tabPanel("Training and Testing",
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
          splitLayout(radioButtons("eta1", label = "eta (learning rate)",
                       choices = c(0.01, 0.05, 0.1, 0.5, 1.0),
                       inline = TRUE),
          radioButtons("epochs1", label =  "epochs",
                       choices = c(10, 100, 1000, 10000),
                       inline = TRUE)),
          plotOutput("activity3a", height = "250px"),
          br(), br(),
            radioButtons("test1", label = "test sample",
                         choices = c("0.6/0.4/0.2", 
                                     "1.0/0.4/0.0",
                                     "0.6/0.2/0.0",
                                     "0.2/0.2/0.2",
                                     "0.6/0.6/0.4"),
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
        img(src = "wrap-up-plot.png", height = "500px")
    )))


) # closing for user interface
