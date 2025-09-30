package main

import (
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

func main() {
	route := gin.Default()
	route.GET("/", func(ctx *gin.Context) {
		logrus.Infof("%+v", ctx.Request.Header)
		ctx.JSON(200, gin.H{"message": "ᕕ( ᐛ )ᕗ"})
	})
	route.GET("/sleep",func(ctx *gin.Context){
		 mode,_  := ctx.GetQuery("mode")
		 sleep_time_str,_ := ctx.GetQuery("sleep_time")
		 if sleep_time_str == ""{
			sleep_time_str = "10"
		 }
		 sleep_time,err := strconv.Atoi(sleep_time_str)
		 if err != nil{
			ctx.JSON(500, gin.H{"message": "参数sleep_time格式错误"})
			return
		}
		time_start := time.Now().Format(time.RFC3339)
		 if mode == ""{
			mode = "random"
		 }
		 if mode == "random"{
			sleep_time = rand.Intn(sleep_time)
		 }
		 time.Sleep(time.Duration(sleep_time) * time.Millisecond)
		 ctx.JSON(200, gin.H{"message": "success","id":ctx.Query("id"),"sleep":sleep_time, "time_start": time_start, "time_end": time.Now().Format(time.RFC3339)})
	})
	args := os.Args[1:]
	host := ":9091"
	if len(args) > 0 {
		host = args[0]
	}
	if err := route.Run(host); err != nil {
		logrus.Fatal(err)
	}
}