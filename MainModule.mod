MODULE MainModule
    VAR string Server_message;
    VAR string client_message;
    VAR string stringX;
    VAR string stringY;
    VAR string stringShape;
    VAR string stringAngle;
    VAR string stringRot;
    
    VAR num numValueX;
    VAR num numValueY;
    VAR num IntShape;
    VAR num Rot;
    VAR num MsgLen;
    VAR num seprationPos;
    VAR num seprationPos_shape;
    VAR num seprationPos_rot;
    
    
    VAR bool okShape;
    VAR bool okXval;
    VAR bool okYval;
    VAR bool okRot;
    

    VAR num rotcalval;
    
	TASK PERS tooldata tool_vacum:=[TRUE,[[-89.8692,3.48376,185.6],[1,0,0,0]],[0.5,[50,0,50],[1,0,0,0],0,0,0]];
    TASK PERS tooldata tool1_cal:=[TRUE,[[-11.2101,11.6913,220.694],[1,0,0,0]],[0.45,[0,0,40],[1,0,0,0],0,0,0]];
    
	TASK PERS wobjdata PickupBay:=[FALSE,TRUE,"",[[630.429,-290.327,-2.15054],[0.987419,0.003592,-0.00345742,0.158044]],[[0,0,0],[1,0,0,0]]];
	TASK PERS wobjdata PlacingBay:=[FALSE,TRUE,"",[[409.288,160.25,5.24787],[0.999996,0.00065811,-0.00167126,0.00205397]],[[0,0,0],[1,0,0,0]]];
	CONST robtarget p10:=[[409.55,-5.21,228.38],[0.0232286,-0.0245904,-0.999407,0.00645006],[-1,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p100:=[[157.03,159.69,50.0],[0.0277802,0.154631,-0.987571,0.00460262],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p200:=[[36.89,38.96,49.98],[0.0279526,0.154742,-0.987548,0.00464847],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p300:=[[152.09,50.23,49.57],[0.0279898,0.15474,-0.987548,0.00465897],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p400:=[[147.17,32.73,50.0],[0.0112201,-0.998484,-0.0531599,-0.00884112],[-1,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget p500:=[[90.98,100.08,50.0],[0.0111998,-0.998484,-0.0531505,-0.00885103],[-1,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    VAR robtarget shape_pos:=[[122.26,129.16,67.28],[0.0217401,-0.824237,-0.565561,-0.0173566],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    VAR robtarget place_pos:=[[55.66,95.99,97.76],[0.0436631,0.243902,-0.968797,0.00615399],[0,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    VAR robtarget place_pos_rot:=[[55.66,95.99,97.76],[0.0436631,0.243902,-0.968797,0.00615399],[0,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
	PROC main()
        MoveL p10,v100,z50, tool_vacum\WObj:=wobj0;
        RobotAsClientConnect;
        client_message:="ready";
                
        WHILE TRUE DO
            RobotClienSendMessage client_message;
            Server_message:=RobotClientReciveMessage();
            TPWrite Server_message;
            MsgLen := StrLen(Server_message);
            seprationPos:= StrFind(Server_message,1,",");
            seprationPos_shape:= StrFind(Server_message,1,"/");
            seprationPos_rot:=StrFind(Server_message,3,"#");
            
            
            stringShape:= StrPart(Server_message,1,(seprationPos_shape-1));             
            stringX := StrPart(Server_message,seprationPos_shape+1,(seprationPos-seprationPos_shape-1));
            stringY := StrPart(Server_message,seprationPos+1,seprationPos_rot-seprationPos-1);
            stringRot:=StrPart(Server_message,(seprationPos_rot+1),(MsgLen-seprationPos_rot-1));
            
            okShape := StrToVal(stringShape,IntShape);
            okXval := StrToVal(stringX,numValueX);
            okYval := StrToVal(stringY,numValueY);
            okRot := StrToVal(stringRot,Rot);
            client_message:="Cordinate recived";
            RobotClienSendMessage client_message;       
            
            client_message:="Cordinate recived";
            RobotClienSendMessage client_message;        
            
            TPWrite "Press DI exicute the new cordinates:";
            WaitDI diOkButton, 1;
            
            placeShape IntShape, numValueX,numValueY,Rot;
            MoveL p10,v100,z50, tool_vacum\WObj:=wobj0;
            client_message:="Done";
            RobotClienSendMessage client_message;
            
            TPWrite "Waiting for new codinates";
        ENDWHILE
        
	ENDPROC
    PROC placeShape(num shape, num x, num y, num rot)
        shape_pos.trans.x:=x;
        shape_pos.trans.y:=y;
        MoveL shape_pos,v100,z50, tool_vacum\WObj:=PickupBay;
        MoveL Offs(shape_pos,0,0,-68),v50,fine,tool_vacum\WObj:=PickupBay;
        SetDO doValve1, 1;
        WaitTime 0.5;
        MoveL shape_pos,v100,z50, tool_vacum\WObj:=PickupBay;
                
    TEST shape
        CASE 0:
        place_pos:=p100;
        CASE 1:
        Rot:=0;
        place_pos:=p200;
        CASE 2:
        place_pos:=p300;
        CASE 3:
        place_pos:=p400;
        CASE 4:
        place_pos:=p500;
        DEFAULT:
        ENDTEST
        
        MoveL place_pos,v100,z50, tool_vacum\WObj:=PlacingBay;
        
        place_pos_rot:= RelTool(place_pos, 0, 0, 0,  \Rz:= Rot);
        MoveL place_pos_rot,v5,fine,tool_vacum\WObj:=PlacingBay;
        
        MoveL Offs(place_pos_rot,0,0,-50),v50,fine,tool_vacum\WObj:=PlacingBay;
        SetDO doValve1, 0;
        WaitTime 0.5;
        MoveL place_pos_rot,v100,z50, tool_vacum\WObj:=PlacingBay;
        
        
        MoveL p10,v100,fine, tool_vacum\WObj:=wobj0;
		
		

    ENDPROC

ENDMODULE