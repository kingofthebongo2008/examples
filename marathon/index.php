<?php

function get_event_name_3( $event )
{
    switch ( $event )
    {
        case 0: return "PIRIN RUN 2014";
        case 1: return "RILA RUN 2014";
        case 2: return "VITOSHA RUN 2014";
        case 3: return "OSOGOVO RUN 2014";
        default:
            die("invalid event name ");
    }
}

function get_event_names2 ( &$events )
{
    $data = array();
    
    array_push ( $data, null );      
    array_push ( $data, null );      
    array_push ( $data, null );      
    array_push ( $data, null );      
    
    foreach( $events as $value )
    {
        $v = intval($value);
        $data [ $v ] = get_event_name_3( $v );
    }
    
    return $data;
}

function get_event_competitor_type( &$checkbox_0, &$checkbox_1, &$checkbox_2, &$checkbox_3 )
{
    $data = array();

    $runner = 0;
    $biker  = 1;
    
    
    array_push ( $data, isset ( $checkbox_0 ) ? $biker : $runner ); 
    array_push ( $data, isset ( $checkbox_1 ) ? $biker : $runner ); 
    array_push ( $data, isset ( $checkbox_2 ) ? $biker : $runner ); 
    array_push ( $data, isset ( $checkbox_3 ) ? $biker : $runner ); 
    
    
    return $data;
}

function get_gender( &$radio_button_0 )
{
    if ( isset($radio_button_0) )
    {
        return intval( $radio_button_0 );   
    }
    else
    {
        return 0;   
    }
}

$name = $_POST["name"];
$surname = $_POST["surname"];
$year_of_birth = intval ( $_POST["year_of_birth"] );

$club = $_POST["club"];
$phone = $_POST["phone"];
$email = $_POST["email"];

$events = $_POST["events"];

$event_names = get_event_names2( $events );


$tmp0 = null;
$tmp1 = null;
$event_type = get_event_competitor_type ( $_POST["event_0_biker"], $tmp0, $tmp1, $_POST["event_3_biker"] );
$gender = get_gender( $_POST["gender"] );

$host="localhost";

$port=3306;
$socket="";
$user="marathon";
$password="cska_moskva_1946";
$dbname="marathon_events";



$con = new mysqli($host, $user, $password, $dbname, $port, $socket)
	or die ('Could not connect to the database server' . mysqli_connect_error());

$con->set_charset('utf8');

$stmt = $con->prepare("insert into events_2014 (event_name, name, surname, year_of_birth, gender, club, phone, email, type ) values ( ?, ?, ?, ?, ?, ?, ?, ?, ? )");

$con->autocommit(false);

for ( $i = 0; $i < 4; ++$i )
{
    if (isset ( $event_names[$i] ) )
    {
       
        $event_name = $event_names[$i];
        $competitor_type = $event_type[$i];
        
        $stmt->bind_param("sssiisssi", $event_name, $name, $surname, $year_of_birth, $gender, $club, $phone, $email, $competitor_type);
        
        if ( ! $stmt->execute() )
        {
            print_r($con->error_list);
        }
    }
}

$con->commit();
$stmt->close();
$con->close();

print("Успешна регистрация / Successfull registration");
?>