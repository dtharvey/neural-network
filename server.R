# server.R file for neural networks

shinyServer(function(input, output, session){

  # images to display for feed-forward tab
  output$activity1a = renderImage({
    display_image = file.path("www",input$image_choice1)
    list(src = display_image, 
         contentType = "image/png", 
         height = 350)}, deleteFile = FALSE)
  
  # images to display on backpropogation tab
  output$activity2a = renderImage({
    display_image = file.path("www",input$image_choice2)
    list(src = display_image, 
         contentType = "image/png", 
         height = 400)}, deleteFile = FALSE)
  
  # code to train network using a single training sample
  trained = reactive({
    out = train(training_samples = as.numeric(input$ts),
                   eta = 10^input$eta1,
                   epochs = 10^input$epochs1)
  })
  
  # code to test trained network using one of four testing samples
  tested = reactive({
    model = trained()

    if (input$test1 == "1/0/1"){
      out = test(model, I1 = 1, I2 = 0, t = 1)
    } else if (input$test1 == "1/1/0"){
      out = test(model, I1 = 1, I2 = 1, t = 0)
    } else if (input$test1 == "0/0/0") {
      out = test(model, I1 = 0, I2 = 0, t = 0)
    } else {
      out = test(model, I1 = 0, I2 = 1, t = 1)
    }
    out
  })
  
  # code to print predictions using test samples
  output$prediction = renderText({
    res = tested()
    paste("Predicted Result is:", round(res$prediction, 4))
  })
  
  # code to print predicted outcome when using test samples
  output$error = renderText({
    res <- tested()
    paste("Error is:", round(res$error, 4))
  }) 
  
  output$activity3a = renderPlot({
    x1 = 1:trained()$steps
    x2 = 1:trained()$epochs
    y1 = trained()$E
    y2 = trained()$E_epoch
    if (input$ts == 1){
    old.par = par(mar = c(5,4,1,2))
    plot(x = x1, y = y1, type = "l", lty = 1, lwd = 3, col = 3,
         xlab = "steps", ylab = "error", ylim = c(0,0.15))
    par(old.par)
    grid()
    }
    
    if (input$ts == 4){
      old.par = par(mar = c(5,4,1,2))
      plot(x = x1, y = y1, type = "l", lty = 1, lwd = 3, col = 3,
           xlab = "steps", ylab = "error")
      if (input$avg == "yes"){
        epoch_steps <- seq(2, trained()$steps, by = 4) 
        lines(x = epoch_steps, y = trained()$E_epoch, col = "red", lwd = 2)
      }
      par(old.par)
      grid()
    }
  })
  
  
}) # keep this to close the server file



