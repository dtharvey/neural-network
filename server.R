# server.R file for neural networks

shinyServer(function(input, output, session){

  # image to display for feed-forward tab
  output$activity1a = renderImage({
    display_image = file.path("www",input$image_choice1)
    list(src = display_image, 
         contentType = "image/png", 
         height = 400)}, deleteFile = FALSE)
  
  # image to display on backpropogation tab
  output$activity2a = renderImage({
    display_image = file.path("www",input$image_choice2)
    list(src = display_image, 
         contentType = "image/png", 
         height = 400)}, deleteFile = FALSE)
  
  # code to train network using a single training sample
  trained = reactive({
    out = train_nn(I1 = 0.6, I2 = 0.4, t = 0.2,
                   eta = as.numeric(input$eta1),
                   epochs = as.numeric(input$epochs1))
  })
  
  # trained = reactive({
  # out = train_test(I1 = rep(0.6, 10000),
  #                  I2 = rep(0.4, 10000),
  #                  t = rep(0.2, 10000),
  #                  eta = 0.05,
  #                  epochs = 10000)
  # })
  
  # trained2 = reactive({
  # 
  #     out = train_nn(I1 = df$input1[index_values[i]],
  #                    I2 = df$input2[index_values[i]],
  #                    t = df$t[index_values[i]],
  #                    eta = as.numeric(input$eta2),
  #                    epochs = as.numeric(input$epochs2))
  #   
  # })
  
  # code to test trained network using one of five testing samples
  tested = reactive({
    model <- trained()
    
    if (input$test1 == "0.6/0.4/0.2"){
      out = test_nn(model, I1 = 0.6, I2 = 0.4, t = 0.2)
    } else if (input$test1 == "1.0/0.4/0.0"){
      out = test_nn(model, I1 = 1.0, I2 = 0.4, t = 0.0)
    } else if (input$test1 == "0.6/0.2/0.0") {
      out = test_nn(model, I1 = 0.6, I2 = 0.2, t = 0.0)
    } else if (input$test1 == "0.2/0.2/0.2"){
      out = test_nn(model, I1 = 0.2, I2 = 0.2, t = 0.2)
    } else {
      out = test_nn(model, I1 = 0.6, I2 = 0.6, t = 0.4)
    }
    out
  })
  
  # code to print predictions using test samples
  output$prediction <- renderText({
    res <- tested()
    paste("Predicted Result is:", round(res$prediction, 4))
  })
  
  # output$prediction2 <- renderText({
  #   res <- tested()
  #   paste("Predicted Result is:", round(res$prediction, 4))
  # })
  
  # code to print predicted outcome when using test samples
  output$error <- renderText({
    res <- tested()
    paste("Error is:", round(res$error, 4))
  }) 
  
  # output$error2 <- renderText({
  #   res <- tested()
  #   paste("Error is:", round(res$error, 4))
  # })
  
  output$activity3a = renderPlot({
    x = 1:trained()$epochs
    y1 = trained()$errors
    old.par = par(mar = c(5,4,1,2))
    plot(x = x, y = y1, type = "l", lty = 1, lwd = 3, col = 3,
         xlab = "epochs", ylab = "error", ylim = c(0,0.08))
    grid()
    par(old.par)
  })
  
  # output$activity4a = renderPlot({
  #   x = 1:trained2()$epochs
  #   y = trained2()$errors
  #   plot(x = x, y = y, type = "l")
  # })
  
}) # keep this to close the server file



